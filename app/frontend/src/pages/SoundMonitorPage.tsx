import { useEffect, useRef, useState } from 'react'
import { Badge } from '../components/ui/badge'
import { MonitorFace } from '../components/MonitorFace'
import { SoundActivity } from '../components/SoundActivity'
import type { MicrophoneMode } from '../components/MicrophoneMonitor'

type Mode = MicrophoneMode

export function SoundMonitorPage() {
  const [mode, setMode] = useState<Mode>('off')
  const [level, setLevel] = useState(0)
  const [soundDetected, setSoundDetected] = useState(false)
  const [error, setError] = useState('')
  const [lastHeard, setLastHeard] = useState('No sound detected yet')
  const streamRef = useRef<MediaStream | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const frameRef = useRef<number | null>(null)
  const modeRef = useRef<Mode>('off')
  const lastSoundRef = useRef(0)
  const soundDetectedRef = useRef(false)

  const changeMode = (next: Mode) => { modeRef.current = next; setMode(next) }

  const stopListening = () => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    frameRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    const context = contextRef.current
    contextRef.current = null
    if (context && context.state !== 'closed') void context.close()
    setLevel(0)
    soundDetectedRef.current = false
    setSoundDetected(false)
    changeMode('off')
  }

  const startListening = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) { setError('Microphone access is not available in this browser.'); return }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const context = new AudioContext()
      contextRef.current = context
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)
      const samples = new Uint8Array(analyser.fftSize)
      changeMode('listening')
      const monitor = () => {
        analyser.getByteTimeDomainData(samples)
        let sum = 0
        for (const sample of samples) { const value = (sample - 128) / 128; sum += value * value }
        const rms = Math.sqrt(sum / samples.length)
        const normalizedLevel = Math.min(1, rms * 8)
        setLevel(normalizedLevel)
        const now = Date.now()
        if (rms > 0.01) {
          lastSoundRef.current = now
          if (!soundDetectedRef.current) {
            soundDetectedRef.current = true
            setSoundDetected(true)
            setLastHeard(new Intl.DateTimeFormat('en', { hour: 'numeric', minute: '2-digit' }).format(now))
          }
          if (modeRef.current === 'waiting') changeMode('listening')
        } else if (soundDetectedRef.current && now - lastSoundRef.current > 1800) {
          soundDetectedRef.current = false
          setSoundDetected(false)
          if (modeRef.current === 'listening') changeMode('waiting')
        }
        frameRef.current = requestAnimationFrame(monitor)
      }
      frameRef.current = requestAnimationFrame(monitor)
    } catch { stopListening(); setError('Microphone access was denied or the device is unavailable.') }
  }

  const toggleMicrophone = () => { if (modeRef.current === 'off') void startListening(); else stopListening() }

  useEffect(() => () => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (contextRef.current && contextRef.current.state !== 'closed') void contextRef.current.close()
  }, [])

  const status = mode === 'off' ? 'Microphone off' : mode === 'waiting' ? 'Waiting for sound' : 'Listening'
  const detail = mode === 'off' ? 'Tap the character to start listening' : mode === 'waiting' ? 'No sound right now · microphone stays on and listening will resume automatically' : 'Listening for nearby sounds · tap the character to turn off'

  return <>
    <section className={`monitor-card mode-${mode}`} aria-label="Sound monitor"><div className="monitor-heading"><div><Badge className={`status-badge badge-${mode}`}><span className="status-dot"/>{status}</Badge><h2>Be here.<br/><span>We’ll listen.</span></h2><p className="monitor-detail">{detail}</p></div><div className="live-indicator"><span/>{mode === 'off' ? 'MONITOR OFF' : mode === 'waiting' ? 'MIC ACTIVE · WAITING' : 'LIVE MONITOR'}</div></div><div className={`mascot-stage stage-${mode}`}><div className="wave-rings"><i/><i/><i/></div><button className="mascot-button" onClick={toggleMicrophone} aria-label={mode === 'off' ? 'Turn microphone on' : 'Turn microphone off'} aria-pressed={mode !== 'off'}><MonitorFace mode={mode}/></button><div className={`sound-pill ${soundDetected ? 'sound-pill-active' : ''}`}><span className="sound-pill-dot"/>{mode === 'off' ? 'Ready when you are' : soundDetected ? 'Sound detected' : mode === 'waiting' ? 'Listening for the next sound' : 'Listening to your space'}</div></div><div className="waveform-panel"><div className="waveform-head"><div><span className="waveform-kicker">SOUND ACTIVITY</span><strong>{mode === 'off' ? 'Your soundscape' : soundDetected ? 'Sound detected' : mode === 'waiting' ? 'Quiet moment' : 'Listening now'}</strong></div><span className="waveform-time">{mode === 'off' ? '—' : 'LIVE'}</span></div><div className={`waveform-bars ${mode === 'off' ? 'bars-off' : mode === 'waiting' ? 'bars-waiting' : ''}`} aria-label="Live sound level visualization">{Array.from({ length: 40 }, (_, index) => { const wave = Math.abs(Math.sin(index * 0.72 + Date.now() / 480)); const barHeight = mode === 'off' ? 7 + wave * 8 : mode === 'waiting' ? 7 + wave * 7 : 8 + Math.max(wave * level * 54, level * 8); return <span key={index} style={{ height: `${barHeight}px` }} /> })}</div><div className="waveform-foot"><span>QUIET</span><span>Live input level · not a recording</span><span>LOUD</span></div></div>{error && <p className="error-message" role="alert">{error}</p>}</section>
    <SoundActivity mode={mode} soundDetected={soundDetected} lastHeard={lastHeard} />
  </>
}
