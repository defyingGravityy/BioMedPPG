import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { io } from 'socket.io-client'
import './App.css'

type GenderOption = 'Male' | 'Female'

type SerialReading = {
  index: number
  red: number
  ir: number
  total: number
  timestamp: number
}

type PredictionStats = {
  total_readings: number
  valid_predictions: number
  outliers_removed: number
  mean_hemoglobin: number
  median_hemoglobin: number
  std_deviation: number
  min_hemoglobin: number
  max_hemoglobin: number
  confidence_95_lower: number
  confidence_95_upper: number
  coefficient_variation: number
  all_predictions?: number[]
}

type DownloadTokens = {
  readings?: string
  results?: string
}

type SessionStatusPayload = {
  sessionId?: string
  status?: string
  simulate?: boolean
}

type SessionCompletePayload = {
  sessionId: string
  summary: string
  stats: PredictionStats
  downloadTokens?: DownloadTokens
}

type SessionErrorPayload = {
  sessionId?: string
  error: string
}

const sanitizeBaseUrl = (value: string) => value.replace(/\/$/, '')
const envBackendUrl = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.trim()
const BACKEND_URL = sanitizeBaseUrl(envBackendUrl && envBackendUrl.length > 0 ? envBackendUrl : 'http://localhost:8000')

const buildDownloadUrl = (baseUrl: string, token?: string) => {
  if (!token) {
    return undefined
  }
  const safeSegments = token
    .split(/[\\/]+/)
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
  if (safeSegments.length === 0) {
    return undefined
  }
  return `${baseUrl}/api/download/${safeSegments.join('/')}`
}

function App() {
  const backendUrl = BACKEND_URL
  const [formState, setFormState] = useState({
    userId: '',
    name: '',
    age: '',
    gender: 'Male' as GenderOption,
    durationSeconds: '60',
    simulate: true,
  })
  const [statusMessage, setStatusMessage] = useState('Idle')
  const [socketConnected, setSocketConnected] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [readingCount, setReadingCount] = useState(0)
  const [lastReading, setLastReading] = useState<SerialReading | null>(null)
  const [readingsPreview, setReadingsPreview] = useState<SerialReading[]>([])
  const [summary, setSummary] = useState('')
  const [stats, setStats] = useState<PredictionStats | null>(null)
  const [downloads, setDownloads] = useState<{ readings?: string; results?: string }>({})
  const [error, setError] = useState<string | null>(null)

  const sessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    const socket = io(backendUrl, {
      transports: ['websocket'],
      autoConnect: true,
    })

    const matchesSession = (incomingSessionId?: string | null) => {
      if (!incomingSessionId) {
        return false
      }
      return incomingSessionId === sessionIdRef.current
    }

    socket.on('connect', () => {
      setSocketConnected(true)
    })

    socket.on('disconnect', () => {
      setSocketConnected(false)
    })

    socket.on('session_status', (payload: SessionStatusPayload) => {
      if (!matchesSession(payload.sessionId)) {
        return
      }
      const simulateLabel = payload.simulate ? 'Simulated readings in progress' : 'Collecting from serial device'
      setStatusMessage(payload.status ? `${payload.status.toUpperCase()} – ${simulateLabel}` : simulateLabel)
      setIsRunning(true)
      setError(null)
    })

    socket.on('serial_reading', (payload: SerialReading & { sessionId?: string }) => {
      if (!matchesSession(payload.sessionId)) {
        return
      }
      const reading: SerialReading = {
        index: payload.index,
        red: payload.red,
        ir: payload.ir,
        total: payload.total,
        timestamp: payload.timestamp,
      }
      setReadingCount(payload.total)
      setLastReading(reading)
      setReadingsPreview((prev) => {
        const next = [...prev, reading]
        return next.slice(-12)
      })
    })

    socket.on('session_complete', (payload: SessionCompletePayload) => {
      if (!matchesSession(payload.sessionId)) {
        return
      }
      sessionIdRef.current = null
      setIsRunning(false)
      setSummary(payload.summary)
      setStats(payload.stats)
      setStatusMessage('Session complete')
      setDownloads({
        readings: buildDownloadUrl(backendUrl, payload.downloadTokens?.readings),
        results: buildDownloadUrl(backendUrl, payload.downloadTokens?.results),
      })
    })

    socket.on('session_error', (payload: SessionErrorPayload) => {
      if (!matchesSession(payload.sessionId)) {
        return
      }
      sessionIdRef.current = null
      setIsRunning(false)
      setError(payload.error || 'Session encountered an error')
      setStatusMessage('Session error')
    })

    return () => {
      socket.removeAllListeners()
      socket.disconnect()
    }
  }, [backendUrl])

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormState((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const resetSessionUiState = () => {
    setStatusMessage('Awaiting first reading…')
    setReadingCount(0)
    setLastReading(null)
    setReadingsPreview([])
    setSummary('')
    setStats(null)
    setDownloads({})
    setError(null)
  }

  const startSession = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (isRunning) {
      setError('A session is already running. Stop it before starting a new one.')
      return
    }

    const userId = formState.userId.trim()
    const name = formState.name.trim()
    const age = Number.parseInt(formState.age, 10)

    if (!userId) {
      setError('Please enter a user ID.')
      return
    }
    if (!name) {
      setError('Please enter a name.')
      return
    }
    if (!Number.isFinite(age) || age <= 0) {
      setError('Please enter a valid age (positive integer).')
      return
    }

    const payload: Record<string, unknown> = {
      userId,
      name,
      age,
      gender: formState.gender,
      simulate: formState.simulate,
    }

    const durationValue = formState.durationSeconds.trim()
    if (durationValue) {
      const duration = Number.parseInt(durationValue, 10)
      if (Number.isFinite(duration) && duration > 0) {
        payload.durationSeconds = duration
      }
    }

    try {
      setStatusMessage('Starting session…')
      setError(null)
      resetSessionUiState()

      const response = await fetch(`${backendUrl}/api/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error || 'Failed to start session')
      }

      const newSessionId = String(data.sessionId)
      sessionIdRef.current = newSessionId
      setSessionId(newSessionId)
      setIsRunning(true)
      setStatusMessage('Awaiting first reading…')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error while starting the session'
      setError(message)
      setIsRunning(false)
      setStatusMessage('Idle')
    }
  }

  const stopSession = async () => {
    const activeSessionId = sessionIdRef.current
    if (!activeSessionId) {
      return
    }
    try {
      await fetch(`${backendUrl}/api/session/${activeSessionId}/stop`, {
        method: 'POST',
      })
      setStatusMessage('Stop requested…')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to stop session'
      setError(message)
    }
  }

  const lastReadingTimestamp = lastReading
    ? new Date(lastReading.timestamp * 1000).toLocaleTimeString()
    : '—'

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Hemoglobin Monitor</h1>
          <p>Stream MAX30102 readings, generate predictions, and download results.</p>
        </div>
        <div className={`connection-indicator ${socketConnected ? 'online' : 'offline'}`}>
          <span className="dot" />
          {socketConnected ? 'Realtime connection ready' : 'Waiting for backend connection'}
        </div>
      </header>

      <main className="app-layout">
        <section className="panel panel-form">
          <h2>Start a new session</h2>
          <form className="session-form" onSubmit={startSession}>
            <div className="field-group">
              <label htmlFor="userId">User ID</label>
              <input
                id="userId"
                name="userId"
                placeholder="e.g. 2023201005"
                value={formState.userId}
                onChange={(event) => handleInputChange('userId', event.target.value)}
                disabled={isRunning}
                required
              />
            </div>

            <div className="field-group">
              <label htmlFor="name">Name</label>
              <input
                id="name"
                name="name"
                placeholder="e.g. Ibrahim"
                value={formState.name}
                onChange={(event) => handleInputChange('name', event.target.value)}
                disabled={isRunning}
                required
              />
            </div>

            <div className="field-row">
              <div className="field-group">
                <label htmlFor="age">Age</label>
                <input
                  id="age"
                  name="age"
                  type="number"
                  min={1}
                  value={formState.age}
                  onChange={(event) => handleInputChange('age', event.target.value)}
                  disabled={isRunning}
                  required
                />
              </div>

              <div className="field-group">
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={formState.gender}
                  onChange={(event) => handleInputChange('gender', event.target.value as GenderOption)}
                  disabled={isRunning}
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label htmlFor="durationSeconds">Session duration (seconds)</label>
                <input
                  id="durationSeconds"
                  name="durationSeconds"
                  type="number"
                  min={1}
                  value={formState.durationSeconds}
                  onChange={(event) => handleInputChange('durationSeconds', event.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="field-group checkbox-group">
                <label htmlFor="simulate">Simulation mode</label>
                <div className="checkbox-wrapper">
                  <input
                    id="simulate"
                    type="checkbox"
                    checked={formState.simulate}
                    onChange={(event) => handleInputChange('simulate', event.target.checked)}
                    disabled={isRunning}
                  />
                  <span>Generate demo readings</span>
                </div>
              </div>
            </div>

            <div className="actions">
              <button type="submit" className="primary" disabled={isRunning}>
                {isRunning ? 'Running…' : 'Start session'}
              </button>
              <button type="button" className="secondary" onClick={stopSession} disabled={!isRunning || !sessionIdRef.current}>
                Stop session
              </button>
            </div>

            {error && <p className="form-error">{error}</p>}
          </form>
        </section>

        <section className="panel panel-status">
          <header className="status-header">
            <div>
              <h2>Status</h2>
              <p>{statusMessage}</p>
            </div>
            <div className="status-meta">
              <span className="badge">Session ID</span>
              <code>{sessionId ?? '—'}</code>
            </div>
          </header>

          <div className="status-grid">
            <div className="status-card">
              <h3>Readings</h3>
              <div className="stat-line">
                <span>Total collected</span>
                <strong>{readingCount}</strong>
              </div>
              <div className="stat-line">
                <span>Last reading time</span>
                <strong>{lastReadingTimestamp}</strong>
              </div>
              {lastReading && (
                <div className="stat-line">
                  <span>Latest values</span>
                  <strong>Red {lastReading.red.toLocaleString()} | IR {lastReading.ir.toLocaleString()}</strong>
                </div>
              )}
            </div>

            <div className="status-card readings-list">
              <h3>Recent readings</h3>
              {readingsPreview.length === 0 ? (
                <p className="placeholder">Waiting for readings…</p>
              ) : (
                <ul>
                  {[...readingsPreview]
                    .slice()
                    .reverse()
                    .map((reading) => (
                      <li key={reading.index}>
                        <span>#{reading.index + 1}</span>
                        <span>{reading.red.toLocaleString()}</span>
                        <span>{reading.ir.toLocaleString()}</span>
                      </li>
                    ))}
                </ul>
              )}
            </div>
          </div>

          <div className="status-card">
            <h3>Prediction summary</h3>
            <pre className="summary-block">
              <code>{summary ? summary : 'Results will appear here once the session finishes.'}</code>
            </pre>

            {stats && (
              <div className="stats-grid">
                <div>
                  <span>Total readings</span>
                  <strong>{stats.total_readings}</strong>
                </div>
                <div>
                  <span>Valid predictions</span>
                  <strong>{stats.valid_predictions}</strong>
                </div>
                <div>
                  <span>Removed outliers</span>
                  <strong>{stats.outliers_removed}</strong>
                </div>
                <div>
                  <span>Mean Hb</span>
                  <strong>{stats.mean_hemoglobin.toFixed(2)} g/dL</strong>
                </div>
                <div>
                  <span>Std dev</span>
                  <strong>{stats.std_deviation.toFixed(3)} g/dL</strong>
                </div>
                <div>
                  <span>95% CI</span>
                  <strong>
                    {stats.confidence_95_lower.toFixed(2)} – {stats.confidence_95_upper.toFixed(2)} g/dL
                  </strong>
                </div>
              </div>
            )}
          </div>

          <div className="status-card downloads">
            <h3>Download results</h3>
            <div className="download-buttons">
              <a
                className="download-button"
                href={downloads.readings}
                download
                target="_blank"
                rel="noopener noreferrer"
                aria-disabled={!downloads.readings}
              >
                Download readings CSV
              </a>
              <a
                className="download-button"
                href={downloads.results}
                download
                target="_blank"
                rel="noopener noreferrer"
                aria-disabled={!downloads.results}
              >
                Download summary TXT
              </a>
            </div>
            {!downloads.readings && <p className="placeholder">Links will appear once predictions are ready.</p>}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
