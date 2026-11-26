| run | model | train_set | best_epoch | val_macro_f1 | test_clean | test_snr20 | test_snr10 | test_snr0 | test_overall |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority_set1 | majority | 1 | - | 0.0642 | 0.0635 | 0.0667 | 0.0667 | 0.0667 | 0.0659 |
| baseline_majority_set2 | majority | 2 | - | 0.0642 | 0.0635 | 0.0667 | 0.0667 | 0.0667 | 0.0659 |
| baseline_majority_set3 | majority | 3 | - | 0.0642 | 0.0635 | 0.0667 | 0.0667 | 0.0667 | 0.0659 |
| baseline_mfcc_lr_set1 | mfcc_lr | 1 | - | 0.5896 | 0.6746 | 0.6666 | 0.6289 | 0.5274 | 0.6245 |
| baseline_mfcc_lr_set2 | mfcc_lr | 2 | - | 0.5862 | 0.6743 | 0.6752 | 0.6415 | 0.5435 | 0.6334 |
| baseline_mfcc_lr_set3 | mfcc_lr | 3 | - | 0.5656 | 0.6608 | 0.6661 | 0.6481 | 0.5586 | 0.6335 |
| deep_aug_set1_seed42 | deep + aug | 1 | 13 | 0.7197 | 0.7048 | 0.6899 | 0.6732 | 0.5209 | 0.6501 |
| deep_aug_set2_seed42 | deep + aug | 2 | 12 | 0.729 | 0.7795 | 0.7865 | 0.7571 | 0.6753 | 0.7499 |
| deep_aug_set3_seed42 | deep + aug | 3 | 13 | 0.7445 | 0.7978 | 0.7996 | 0.7883 | 0.7169 | 0.7757 |
| deep_set2_seed42 | deep | 2 | 8 | 0.71 | 0.77 | 0.7636 | 0.746 | 0.6182 | 0.7249 |
| small_set2_seed42 | small | 2 | 49 | 0.6133 | 0.668 | 0.6862 | 0.6568 | 0.5129 | 0.6314 |
