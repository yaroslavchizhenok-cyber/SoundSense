import type { MicrophoneMode } from './MicrophoneMonitor'

type SoundActivityProps = {
  mode: MicrophoneMode
  soundDetected: boolean
  lastHeard: string
}

export function SoundActivity({ mode, soundDetected, lastHeard }: SoundActivityProps) {
  const title = mode === 'off' ? 'Your space is quiet' : soundDetected ? 'We heard something' : mode === 'waiting' ? 'A quiet moment' : 'Listening for sounds'
  const detail = mode === 'off' ? 'Turn on your microphone to see sound activity here.' : `Last sound detected: ${soundDetected ? lastHeard : '—'}`
  return (
    <aside className="activity-card">
      <div className="activity-title"><div><p className="eyebrow">YOUR SPACE</p><h3>Sound activity</h3></div><span className="activity-spark">✳</span></div>
      <div className="activity-empty"><div className="activity-orb"><span>〰</span></div><strong>{title}</strong><p>{detail}</p></div>
      <div className="activity-note"><span>ⓘ</span><p>SoundSense detects sound levels only. It does not record audio or create a transcript.</p></div>
      <div className="activity-legend"><span><i className="legend-green"/>Sound detected</span><span><i className="legend-soft"/>Quiet</span></div>
    </aside>
  )
}
