"""Train one SoundSense CNN (Part C).

Examples (run from inside the SoundSense folder):
    python train.py --model small --train-set 2                # M1
    python train.py --model deep  --train-set 2                # M2
    python train.py --model deep  --train-set 2 --augment      # M3

Everything except --model, --train-set and --augment has a fixed default. Keep the
defaults identical across runs: data_manual rule 5 (fair comparison) requires the same
seed, optimizer, learning rate, batch size, augmentation and stopping rule.

Output: runs/<run name>/
    best.pth          weights of the best epoch + what the app needs (stats, class order)
    history.csv       loss and macro-F1 per epoch
    curves.png        loss curves + validation macro-F1
    summary.json      settings and best validation score
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data
from model import build_model, count_parameters

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train a SoundSense CNN")
    p.add_argument("--model", choices=["small", "deep"], required=True)
    p.add_argument("--train-set", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--augment", action="store_true", help="time shift + SpecAugment (M3)")
    p.add_argument("--epochs", type=int, default=50, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better validation macro-F1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(data.DEFAULT_DATA_DIR), help="generated audio folder")
    p.add_argument("--cache-dir", default=str(data.DEFAULT_CACHE_DIR), help="spectrogram cache folder")
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    p.add_argument("--limit", type=int, default=0,
                   help="debug only: use the first N training clips (e.g. 50 for the overfit test)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(args):
    name = f"{args.model}{'_aug' if args.augment else ''}_set{args.train_set}_seed{args.seed}"
    return name + (f"_limit{args.limit}" if args.limit else "")


def train_one_epoch(model, loader, loss_fn, optimizer, device, use_augment):
    model.train()                                  # Dropout on, BatchNorm learning
    total_loss = 0.0
    for X, y in loader:
        if use_augment:
            X = data.augment(X)                    # training batches only
        X, y = X.to(device), y.to(device)
        logits = model(X)                          # 1. forward pass
        loss = loss_fn(logits, y)                  # 2. loss
        optimizer.zero_grad()                      # 3. reset gradients
        loss.backward()                            # 4. backpropagation
        optimizer.step()                           # 5. update weights
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()                                   # Dropout off, BatchNorm frozen
    total_loss, preds, labels = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, y).item() * X.size(0)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(len(data.CLASSES))),
                        zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def plot_curves(history, out_path, title):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Validation loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss"); ax1.legend()
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], color="tab:green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation macro-F1"); ax2.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = run_name(args)
    out_dir = Path(args.runs_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {name} | device: {device}")

    # ---------- data ----------
    X_train, y_train, _ = data.load_list(f"trainSet{args.train_set}", args.data_dir, args.cache_dir)
    X_val, y_val, _ = data.load_list("valSet", args.data_dir, args.cache_dir)
    if args.limit:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    mean, std, stats_name = data.load_stats(args.train_set)   # stats of THIS training set
    X_train, X_val = data.scale(X_train, mean, std), data.scale(X_val, mean, std)
    y_train, y_val = torch.from_numpy(y_train), torch.from_numpy(y_val)
    print(f"Train {tuple(X_train.shape)} | Val {tuple(X_val.shape)} | stats {stats_name}")

    g = torch.Generator().manual_seed(args.seed)                # same shuffling order every run
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

    # ---------- model, loss, optimizer ----------
    model = build_model(args.model, n_classes=len(data.CLASSES)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Model {args.model}: {count_parameters(model):,} parameters | augment: {args.augment}")

    # ---------- training with early stopping on validation macro-F1 ----------
    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.augment)
        val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1})

        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.model,
                "classes": data.CLASSES,
                "stats_file": stats_name,
                "mean": mean,
                "std": std,
                "input_shape": [1, data.N_MELS, data.N_FRAMES],
                "train_set": args.train_set,
                "augment": args.augment,
                "epoch": epoch,
                "val_macro_f1": val_f1,
            }, out_dir / "best.pth")
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {val_f1:.4f}{'  *best*' if improved else ''} | {time.time() - t0:.0f}s")
        if bad_epochs >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs")
            break

    # ---------- save the record of this run ----------
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, out_dir / "curves.png", name)
    summary = {
        "run": name, "model": args.model, "augment": args.augment, "train_set": args.train_set,
        "stats_file": stats_name, "seed": args.seed, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "patience": args.patience, "limit": args.limit,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "parameters": count_parameters(model), "device": device,
        "minutes": round((time.time() - start) / 60, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest val macro-F1 {best_f1:.4f} at epoch {best_epoch}. Saved to {out_dir}")


if __name__ == "__main__":
    main()