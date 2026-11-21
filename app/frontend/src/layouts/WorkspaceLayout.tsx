import { Sidebar } from '../components/Sidebar'
import { Button } from '../components/ui/button'
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
          <Button variant="outline" className="profile-button"><span className="profile-avatar">S</span>My space</Button>
        </header>
        <div className="content-grid"><Outlet /></div>
        <footer className="workspace-footer"><span>SoundSense <b>·</b> Sound awareness, made simple</span><span><i/>Private by design</span></footer>
      </section>
    </main>
  )
}