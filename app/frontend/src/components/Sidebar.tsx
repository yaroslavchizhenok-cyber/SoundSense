import { Button } from './ui/button'
import { useNavigate } from 'react-router-dom'

export function Sidebar() {
  const navigate = useNavigate()

  return (
    <aside className="sidebar">
      <a className="brand" href="#home" aria-label="SoundSense home">
        <img src="/logo.svg" alt="" />
        <span>Sound<span>Sense</span></span>
      </a>
      <div className="sidebar-label">WORKSPACE</div>
      <Button variant="ghost" className="nav-item nav-active" onClick={() => navigate('/')}>
        <span className="nav-dot"/>Sound monitor
      </Button>
      <div className="sidebar-bottom">
        <div className="privacy-card">
          <span className="privacy-icon">✳</span>
          <strong>Your sound stays yours</strong>
          <p>Audio is analyzed on this device and is not recorded or sent.</p>
        </div>
      </div>
    </aside>
  )
}