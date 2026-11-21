export type MicrophoneMode = 'off' | 'listening' | 'waiting'

type MicrophoneMonitorProps = {
  mode: MicrophoneMode
  level: number
  soundDetected: boolean
  error: string
  lastHeard: string
  onToggle: () => void
}

export function MicrophoneMonitor({ mode, level, soundDetected, error, lastHeard, onToggle }: MicrophoneMonitorProps) {
  const isOn = mode !== 'off'
  const status = mode === 'off' ? 'Microphone off' : mode === 'waiting' ? 'Waiting for sound' : soundDetected ? 'Sound detected' : 'Listening'
  const description = mode === 'off' ? 'Tap the character to start listening' : mode === 'waiting' ? 'The microphone stays active and will resume when sound returns' : 'Your microphone is active · sound is analyzed on this device'
  const bars = Array.from({ length: 36 }, (_, index) => {
    const activity = Math.max(0.12, level * (0.48 + Math.abs(Math.sin(index * 1.7)) * 0.9))
    return <i key={index} style={{ height: isOn ? String(Math.min(100, activity * 100)) + '%' : '9px', width: '6px', display: 'block', borderRadius: '999px', background: '#1cdf77', flexShrink: 0 }} />
  })

  return (
    <section className={'monitor-card ' + (isOn ? 'monitor-on' : 'monitor-off')} aria-label="Sound monitor">
      <div className="monitor-heading">
        <div>
          <p className="eyebrow">YOUR PERSONAL SOUND SPACE</p>
          <h2>Be here. We’ll listen.</h2>
          <p className="monitor-description">{description}</p>
        </div>
        <span className={'status-badge ' + (isOn ? 'status-active' : 'status-idle')}><i />{status}</span>
      </div>
      <button type="button" className={'face-button ' + (isOn ? 'face-listening' : 'face-off')} onClick={onToggle} aria-label={isOn ? 'Turn microphone off' : 'Turn microphone on'} aria-pressed={isOn} style={{ display: 'flex', width: '100%', minHeight: '470px', alignItems: 'center', justifyContent: 'center', padding: 0, border: 0, background: 'transparent', boxShadow: 'none', cursor: 'pointer', overflow: 'visible' }}>
        <svg className="sound-character" width="430" height="460" style={{ display: 'block', width: 'min(430px, 90%)', height: 'auto', flexShrink: 0, opacity: 1, visibility: 'visible', overflow: 'visible' }} viewBox="0 0 280 300" role="img" aria-label={isOn ? 'SoundSense is listening' : 'SoundSense is off'}>
          <ellipse cx="33" cy="164" rx="25" ry="42" fill="#70ad4b" />
          <ellipse cx="247" cy="164" rx="25" ry="42" fill="#70ad4b" />
          <ellipse className="face-shape" cx="140" cy="165" rx="94" ry="112" fill="#c8f27d" stroke="#a9e947" strokeWidth="2" />
          <path d="M83 83c20-29 53-43 83-45" fill="none" stroke="#eaffb8" strokeWidth="7" strokeLinecap="round" />
          <circle cx="91" cy="178" r="9" fill="#f49b78" />
          <circle cx="189" cy="178" r="9" fill="#f49b78" />
          <g className="headphones" style={{ opacity: isOn ? 1 : 0, transform: isOn ? 'translateY(0)' : 'translateY(-12px)', transition: 'opacity 220ms ease, transform 220ms ease' }}>
            <path d="M49 163C48 25 232 25 231 163" fill="none" stroke="#344b40" strokeWidth="12" strokeLinecap="round" />
            <rect x="41" y="140" width="27" height="61" rx="13" fill="#344b40" />
            <rect x="212" y="140" width="27" height="61" rx="13" fill="#344b40" />
            <rect x="50" y="151" width="9" height="37" rx="5" fill="#1cdf77" />
            <rect x="221" y="151" width="9" height="37" rx="5" fill="#1cdf77" />
          </g>
          <g className="eyes-open" style={{ opacity: isOn ? 0 : 1, transform: isOn ? 'scaleY(0.65)' : 'scaleY(1)', transformOrigin: 'center', transition: 'opacity 180ms ease, transform 180ms ease' }}>
            <g className="eye-shell">
              <ellipse className="eye-white" cx="111" cy="151" rx="18" ry="25" fill="#fff" stroke="#344b40" strokeWidth="3"><animate attributeName="ry" values="25;25;2;25;25" keyTimes="0;0.44;0.48;0.52;1" dur="5s" repeatCount="indefinite" /></ellipse>
              <g className="eye-pupil-group"><animateTransform attributeName="transform" type="translate" values="0 0;3 0;0 1;-3 0;0 0" keyTimes="0;0.25;0.5;0.75;1" dur="6s" repeatCount="indefinite" />
                <ellipse className="eye-pupil" cx="111" cy="158" rx="13" ry="18" fill="#080a09"><animate attributeName="ry" values="18;18;1;18;18" keyTimes="0;0.44;0.48;0.52;1" dur="5s" repeatCount="indefinite" /></ellipse>
                <circle cx="118" cy="149" r="6" fill="#fff" /><circle cx="106" cy="164" r="2.5" fill="#fff" />
              </g>
            </g>
            <g className="eye-shell">
              <ellipse className="eye-white" cx="169" cy="151" rx="18" ry="25" fill="#fff" stroke="#344b40" strokeWidth="3"><animate attributeName="ry" values="25;25;2;25;25" keyTimes="0;0.44;0.48;0.52;1" dur="5s" repeatCount="indefinite" /></ellipse>
              <g className="eye-pupil-group"><animateTransform attributeName="transform" type="translate" values="0 0;3 0;0 1;-3 0;0 0" keyTimes="0;0.25;0.5;0.75;1" dur="6s" repeatCount="indefinite" />
                <ellipse className="eye-pupil" cx="169" cy="158" rx="13" ry="18" fill="#080a09"><animate attributeName="ry" values="18;18;1;18;18" keyTimes="0;0.44;0.48;0.52;1" dur="5s" repeatCount="indefinite" /></ellipse>
                <circle cx="176" cy="149" r="6" fill="#fff" /><circle cx="164" cy="164" r="2.5" fill="#fff" />
              </g>
            </g>
          </g>
          <g className="eyes-closed" style={{ opacity: isOn ? 1 : 0, transform: isOn ? 'scaleY(1)' : 'scaleY(0.65)', transformOrigin: 'center', transition: 'opacity 180ms ease, transform 180ms ease' }} fill="none" stroke="#173f31" strokeWidth="6" strokeLinecap="round"><path d="M94 153q17-18 34 0" /><path d="M152 153q17-18 34 0" /></g>
          <path d="M119 192q21 17 42 0" fill="none" stroke="#173f31" strokeWidth="7" strokeLinecap="round" />
        </svg>
      </button>
      <div className="face-caption" style={{ position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px', width: 'fit-content', margin: '-26px auto 16px', padding: '12px 20px', borderRadius: '999px', background: '#fff', border: '1px solid #e1ebd9', boxShadow: '0 12px 30px rgba(31, 89, 58, 0.1)' }}><span className="status-pulse" />{mode === 'off' ? 'Ready when you are' : mode === 'waiting' ? 'Waiting for sound · microphone remains active' : 'Listening to your space'}</div>
      <div className="live-monitor">
        <div className="live-monitor-heading"><div><p className="eyebrow">LIVE MONITOR</p><strong>{soundDetected ? 'Sound detected' : mode === 'waiting' ? 'Quiet moment' : 'Your soundscape'}</strong></div><span>{mode === 'off' ? 'OFF' : 'LIVE'}</span></div>
        <div className={'waveform ' + (isOn ? 'waveform-active' : '')} aria-label="Live sound level visualization" style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '5px', width: '100%', height: '76px', padding: '8px 0' }}>{bars}</div>
        <div className="waveform-labels"><span>QUIET</span><span>{mode === 'waiting' ? 'Microphone active · waiting for sound' : 'Live input level · not a recording'}</span><span>LOUD</span></div>
        {error && <p className="monitor-error" role="alert">{error}</p>}
        {soundDetected && <p className="last-heard">Last sound detected at {lastHeard}</p>}
      </div>
    </section>
  )
}
