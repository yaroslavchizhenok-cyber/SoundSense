import { Sidebar } from '../components/Sidebar'
import { Outlet } from 'react-router-dom'

export function WorkspaceLayout() {
  return (
    <main className="app-shell">
      <Sidebar />
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">YOUR PERSONAL SOUND SPACE</p>
            <h1>Sound monitor</h1>
          </div>
        </header>
        <div className="content-grid"><Outlet /></div>
        <footer className="workspace-footer"><span>SoundSense <b>·</b> Sound awareness, made simple</span><span><i/>Private by design</span></footer>
      </section>
    </main>
  )
}