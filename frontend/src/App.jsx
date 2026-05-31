import { lazy, startTransition, Suspense, useEffect, useRef, useState } from 'react'
import './App.css'
import {
  analyzeRun,
  createRun,
  fetchBrainFrames,
  fetchBrainMesh,
  fetchHealth,
  fetchJobs,
  fetchRunDetail,
  fetchRuns,
} from './api'
import { getZoneColor } from './zonePalette'

const ThreeBrainViewer = lazy(() =>
  import('./components/ThreeBrainViewer').then((module) => ({
    default: module.ThreeBrainViewer,
  })),
)

const DEFAULT_UPLOAD_SETTINGS = {
  checkpoint: 'facebook/tribev2',
  device: 'cuda',
  num_workers: 0,
  text_model_name: 'meta-llama/Llama-3.2-3B',
  text_mode: 'paper',
  transcribe: false,
  staggered_sampling: false,
  seconds_per_word: 0.45,
  max_context_words: 128,
}

const DEFAULT_ANALYSIS_SETTINGS = {
  api_key: '',
  model: 'gpt-5.4',
  reasoning_effort: 'medium',
  image_detail: 'low',
  max_images: 4,
}

const EMPTY_TIMELINE = []
const ACTIVATION_GRADIENT =
  'linear-gradient(to top, #000004 0%, #1b0c41 12%, #4a0c6b 25%, #781c6d 38%, #a52c60 50%, #cf4446 62%, #ed6925 74%, #fb9b06 86%, #fcfdbf 100%)'
const BRAIN_OVERLAY_OPTIONS = [
  { value: 'hcp', label: 'HCP-MMP boundaries' },
  { value: 'systems', label: 'Broad systems' },
  { value: 'none', label: 'None' },
]

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function formatSeconds(seconds) {
  const total = Math.max(0, Number(seconds || 0))
  const minutes = Math.floor(total / 60)
  const remaining = total - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${remaining.toFixed(1).padStart(4, '0')}`
}

function findTimestepIndex(timeline, currentTime) {
  if (!timeline?.length) {
    return 0
  }
  for (let index = timeline.length - 1; index >= 0; index -= 1) {
    if (currentTime >= timeline[index].start_s) {
      return index
    }
  }
  return 0
}

function buildLinePath(values, width, height, padding, maxValue) {
  if (!values.length) {
    return ''
  }
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom
  return values
    .map((value, index) => {
      const x =
        padding.left +
        (values.length === 1 ? innerWidth / 2 : (index / (values.length - 1)) * innerWidth)
      const y =
        padding.top + innerHeight - (Math.max(value, 0) / Math.max(maxValue, 0.001)) * innerHeight
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

function ZoneChart({ series, selectedTimestep, onSelect }) {
  const width = 980
  const height = 320
  const padding = { top: 18, right: 20, bottom: 34, left: 44 }
  const maxValue = Math.max(
    0.001,
    ...series.flatMap((entry) => entry.values ?? []),
  )
  const maxLength = Math.max(1, ...series.map((entry) => entry.values?.length ?? 0))
  const guideX =
    padding.left +
    (maxLength <= 1
      ? (width - padding.left - padding.right) / 2
      : (selectedTimestep / (maxLength - 1)) * (width - padding.left - padding.right))

  const handlePointer = (event) => {
    if (!onSelect || maxLength <= 1) {
      return
    }
    const bounds = event.currentTarget.getBoundingClientRect()
    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1)
    onSelect(Math.round(ratio * (maxLength - 1)))
  }

  return (
    <div className="zone-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="zone-chart__svg"
        onClick={handlePointer}
        onMouseMove={(event) => {
          if (event.buttons === 1) {
            handlePointer(event)
          }
        }}
      >
        <rect x="0" y="0" width={width} height={height} rx="20" className="zone-chart__backdrop" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y =
            padding.top + (height - padding.top - padding.bottom) * (1 - tick)
          return (
            <g key={tick}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                className="zone-chart__grid"
              />
              <text x="12" y={y + 4} className="zone-chart__axis">
                {(tick * maxValue).toFixed(2)}
              </text>
            </g>
          )
        })}
        <line
          x1={guideX}
          y1={padding.top}
          x2={guideX}
          y2={height - padding.bottom}
          className="zone-chart__guide"
        />
        {series.map((entry) => (
          <path
            key={entry.zone_key ?? entry.zone}
            d={buildLinePath(entry.values, width, height, padding, maxValue)}
            fill="none"
            stroke={getZoneColor(entry.zone_key)}
            strokeWidth="3"
            strokeLinecap="round"
            className="zone-chart__line"
          />
        ))}
      </svg>
      <div className="zone-chart__legend">
        {series.map((entry) => (
          <div key={entry.zone_key ?? entry.zone} className="zone-chart__legend-item">
            <span
              className="zone-chart__swatch"
              style={{ background: getZoneColor(entry.zone_key) }}
            />
            <span>{entry.zone}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricCard({ label, value, accent }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong style={accent ? { color: accent } : undefined}>{value}</strong>
    </div>
  )
}

function ActivationScaleCard({ displayMeta }) {
  const range = displayMeta?.activation_range ?? [0, 1]
  const units = displayMeta?.activation_units ?? 'normalized activation'
  const normalization = displayMeta?.normalization ?? 'shared_percentile_99_reference'

  return (
    <div className="brain-display-card">
      <span className="brain-display-card__label">Activation Scale</span>
      <div className="brain-scale">
        <div className="brain-scale__bar" style={{ background: ACTIVATION_GRADIENT }} />
        <div className="brain-scale__ticks">
          <strong>{Number(range[1] ?? 1).toFixed(1)}</strong>
          <span>{units}</span>
          <strong>{Number(range[0] ?? 0).toFixed(1)}</strong>
        </div>
      </div>
      <p className="brain-display-card__note">
        Quantitative color only. Normalization: <code>{normalization}</code>. During playback,
        the 3D view interpolates between adjacent timesteps for continuity, and the translucent
        spread is a presentation layer rather than extra signal.
      </p>
    </div>
  )
}

function BrainOverlayCard({ overlayMode, onOverlayChange, mesh, displayMeta }) {
  return (
    <div className="brain-display-card">
      <span className="brain-display-card__label">Overlay</span>
      <label className="brain-overlay__control">
        <span>Guide layer</span>
        <select value={overlayMode} onChange={(event) => onOverlayChange(event.target.value)}>
          {BRAIN_OVERLAY_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      {overlayMode === 'hcp' ? (
        <p className="brain-display-card__note">
          Thin parcel boundaries from the HCP-MMP atlas. Overlay does not alter activation values.
        </p>
      ) : null}
      {overlayMode === 'systems' ? (
        <>
          <p className="brain-display-card__note">
            Broad cortical systems are pedagogical guides, not canonical network labels.
          </p>
          <div className="brain-overlay__legend">
            {(mesh?.zone_keys ?? []).map((zoneKey) => (
              <div key={zoneKey} className="brain-overlay__legend-item">
                <span
                  className="brain-overlay__swatch"
                  style={{ background: getZoneColor(zoneKey) }}
                />
                <span>{mesh?.zone_labels?.[zoneKey] ?? zoneKey}</span>
              </div>
            ))}
          </div>
        </>
      ) : null}
      {overlayMode === 'none' ? (
        <p className="brain-display-card__note">
          {displayMeta?.overlay_note ?? 'No categorical guide overlay is active.'}
        </p>
      ) : null}
    </div>
  )
}

function TimelineScroller({
  timeline,
  selectedTimestep,
  onSelect,
  isPlaying = false,
  onTogglePlay = null,
  className = '',
}) {
  if (!timeline.length) {
    return null
  }

  const activeStep = timeline[selectedTimestep] ?? timeline[0]

  return (
    <div className={`timeline-card ${className}`.trim()}>
      <div className="timeline-card__controls">
        {onTogglePlay ? (
          <button type="button" className="button timeline-card__button" onClick={onTogglePlay}>
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        ) : null}
        <input
          type="range"
          min="0"
          max={Math.max(0, timeline.length - 1)}
          value={selectedTimestep}
          onChange={(event) => onSelect(Number(event.target.value))}
        />
      </div>
      <div className="timeline-card__meta">
        <strong>{activeStep?.text || 'No aligned transcript for this timestep.'}</strong>
        <span>
          Timestep t{activeStep?.timestep ?? selectedTimestep} · Start{' '}
          {formatSeconds(activeStep?.start_s ?? 0)} · Window{' '}
          {Number(activeStep?.duration_s ?? 0).toFixed(2)}s
        </span>
      </div>
    </div>
  )
}

function RunLibrary({ runs, activeRunId, onSelect }) {
  return (
    <div className="library-grid">
      {runs.map((run) => (
        <button
          type="button"
          key={run.id}
          className={`library-card ${activeRunId === run.id ? 'library-card--active' : ''}`}
          onClick={() => {
            startTransition(() => onSelect(run.id))
          }}
        >
          {run.preview_url ? (
            <img src={run.preview_url} alt="" className="library-card__image" />
          ) : (
            <div className="library-card__placeholder">{run.input_kind}</div>
          )}
          <div className="library-card__body">
            <strong>{run.title}</strong>
            <span>{run.subtitle}</span>
            <small>{run.timesteps} timesteps</small>
          </div>
        </button>
      ))}
    </div>
  )
}

function JobRail({ jobs }) {
  if (!jobs.length) {
    return <p className="panel-note">No recent processing jobs.</p>
  }
  return (
    <div className="job-list">
      {jobs.slice(0, 6).map((job) => (
        <div key={job.id} className={`job-card job-card--${job.status}`}>
          <div>
            <strong>{job.source_name}</strong>
            <span>{job.progress_label}</span>
          </div>
          <div className="job-card__meta">
            <small>{job.status}</small>
            <small>{job.progress_pct}%</small>
          </div>
        </div>
      ))}
    </div>
  )
}

function ChatPanel({
  activeRunId,
  analysisSettings,
  onSettingsChange,
  messages,
  prompt,
  onPromptChange,
  onSubmit,
  busy,
}) {
  return (
    <section className="panel">
      <div className="panel__head">
        <div>
          <p className="panel__kicker">AI interpretation</p>
          <h2>Optional run analysis</h2>
        </div>
      </div>
      <div className="analysis-settings">
        <label>
          OpenAI API key
          <input
            type="password"
            value={analysisSettings.api_key}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                api_key: event.target.value,
              }))
            }
            placeholder="sk-..."
          />
        </label>
        <div className="analysis-settings__row">
          <label>
            Chat model
            <input
              value={analysisSettings.model}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  model: event.target.value,
                }))
              }
            />
          </label>
          <label>
            Reasoning effort
            <select
              value={analysisSettings.reasoning_effort}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  reasoning_effort: event.target.value,
                }))
              }
            >
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
              <option value="xhigh">xhigh</option>
            </select>
          </label>
          <label>
            Panel image detail
            <select
              value={analysisSettings.image_detail}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  image_detail: event.target.value,
                }))
              }
            >
              <option value="low">low</option>
              <option value="auto">auto</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            Timesteps sent
            <input
              type="number"
              min="1"
              max="6"
              value={analysisSettings.max_images}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  max_images: Number(event.target.value || 4),
                }))
              }
            />
          </label>
        </div>
      </div>
      <div className="chat-log">
        {messages.length === 0 ? (
          <p className="panel-note">
            Ask for a second-pass interpretation of the selected TRIBE run. This sends run
            statistics and selected cortical panel images to the configured chat model.
          </p>
        ) : null}
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`chat-bubble chat-bubble--${message.role}`}
          >
            <span>{message.role}</span>
            <p>{message.content}</p>
          </div>
        ))}
      </div>
      <form
        className="chat-form"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit(activeRunId)
        }}
      >
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder="Interpret the peak activation window and identify the dominant cortical systems."
          rows={4}
        />
        <button type="submit" className="button button--primary" disabled={busy || !activeRunId}>
          {busy ? 'Analyzing...' : 'Analyze run'}
        </button>
      </form>
    </section>
  )
}

function App() {
  const [health, setHealth] = useState(null)
  const [runs, setRuns] = useState([])
  const [jobs, setJobs] = useState([])
  const [mesh, setMesh] = useState(null)
  const [brainFrames, setBrainFrames] = useState(null)
  const [runDetail, setRunDetail] = useState(null)
  const [activeRunId, setActiveRunId] = useState(null)
  const [activeJob, setActiveJob] = useState(null)
  const [selectedTimestep, setSelectedTimestep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [brainOverlayMode, setBrainOverlayMode] = useState('hcp')
  const [busy, setBusy] = useState(false)
  const [loadingRun, setLoadingRun] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState('')
  const [uploadSettings, setUploadSettings] = useState(DEFAULT_UPLOAD_SETTINGS)
  const [analysisSettings, setAnalysisSettings] = useState(DEFAULT_ANALYSIS_SETTINGS)
  const [chatMessages, setChatMessages] = useState([])
  const [chatPrompt, setChatPrompt] = useState('')
  const [chatBusy, setChatBusy] = useState(false)
  const [responseId, setResponseId] = useState(null)
  const videoRef = useRef(null)
  const syncGuardRef = useRef(0)
  const meshRef = useRef(null)
  const uploadPreviewUrlRef = useRef('')
  const playbackStateRef = useRef({ time: 0, isPlaying: false })
  const activeJobId = activeJob?.id ?? null

  const activeFrame = brainFrames?.frames?.[selectedTimestep] ?? null
  const timeline = runDetail?.timeline ?? EMPTY_TIMELINE

  useEffect(() => {
    let cancelled = false
    async function prime() {
      try {
        const [healthResponse, runsResponse, jobsResponse] = await Promise.all([
          fetchHealth(),
          fetchRuns(),
          fetchJobs(),
        ])
        if (cancelled) {
          return
        }
        setHealth(healthResponse)
        setRuns(runsResponse.items ?? [])
        setJobs(jobsResponse.items ?? [])
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError.message)
        }
      }
    }
    prime()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (activeRunId || !runs.length) {
      return
    }
    startTransition(() => {
      setActiveRunId(runs[0].id)
    })
  }, [activeRunId, runs])

  useEffect(() => {
    let cancelled = false
    let timeoutId = null

    async function poll() {
      try {
        const [runsResponse, jobsResponse] = await Promise.all([fetchRuns(), fetchJobs()])
        if (cancelled) {
          return
        }
        setRuns(runsResponse.items ?? [])
        setJobs(jobsResponse.items ?? [])

        if (activeJobId) {
          const currentJob = (jobsResponse.items ?? []).find((job) => job.id === activeJobId)
          if (currentJob) {
            setActiveJob(currentJob)
            if (currentJob.status === 'completed' && currentJob.saved_run_id) {
              setNotice(`Run ready: ${currentJob.saved_run_id}`)
              setError('')
              setActiveJob(null)
              startTransition(() => {
                setActiveRunId(currentJob.saved_run_id)
              })
              return
            }
            if (currentJob.status === 'failed') {
              setError(currentJob.error || 'Processing failed.')
              setActiveJob(null)
            }
          }
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError.message)
        }
      } finally {
        timeoutId = window.setTimeout(poll, activeJobId ? 1500 : 5000)
      }
    }

    poll()
    return () => {
      cancelled = true
      if (timeoutId) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [activeJobId])

  useEffect(() => {
    if (!activeRunId) {
      return undefined
    }
    let cancelled = false

    async function loadRun(runId) {
      setLoadingRun(true)
      setError('')
      try {
        const cachedMesh = meshRef.current
        const [detailResponse, framesResponse, meshResponse] = await Promise.all([
          fetchRunDetail(runId),
          fetchBrainFrames(runId),
          cachedMesh ? Promise.resolve(cachedMesh) : fetchBrainMesh(),
        ])
        if (cancelled) {
          return
        }
        startTransition(() => {
          setRunDetail(detailResponse)
          setBrainFrames(framesResponse)
          if (!cachedMesh) {
            meshRef.current = meshResponse
            setMesh(meshResponse)
          }
          setSelectedTimestep(0)
          setIsPlaying(false)
          playbackStateRef.current = { time: 0, isPlaying: false }
          setChatMessages([])
          setChatPrompt('')
          setResponseId(null)
          setAnalysisSettings((current) => ({
            ...current,
            ...(detailResponse.openai_defaults ?? {}),
          }))
        })
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError.message)
        }
      } finally {
        if (!cancelled) {
          setLoadingRun(false)
        }
      }
    }

    loadRun(activeRunId)
    return () => {
      cancelled = true
    }
  }, [activeRunId])

  useEffect(() => {
    return () => {
      if (uploadPreviewUrlRef.current) {
        URL.revokeObjectURL(uploadPreviewUrlRef.current)
      }
    }
  }, [])

  function handleUploadFileChange(file) {
    if (uploadPreviewUrlRef.current) {
      URL.revokeObjectURL(uploadPreviewUrlRef.current)
    }
    const nextUrl = file ? URL.createObjectURL(file) : ''
    uploadPreviewUrlRef.current = nextUrl
    setUploadFile(file)
    setUploadPreviewUrl(nextUrl)
  }

  useEffect(() => {
    const video = videoRef.current
    if (!video) {
      return
    }
    if (isPlaying) {
      playbackStateRef.current.isPlaying = true
      video.play().catch(() => {})
    } else {
      playbackStateRef.current.isPlaying = false
      video.pause()
    }
  }, [isPlaying, runDetail?.source_url])

  useEffect(() => {
    if (!isPlaying) {
      return undefined
    }
    let frameId = 0
    const tick = () => {
      playbackStateRef.current.time =
        videoRef.current?.currentTime ?? playbackStateRef.current.time
      frameId = window.requestAnimationFrame(tick)
    }
    tick()
    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [isPlaying, runDetail?.source_url])

  useEffect(() => {
    const video = videoRef.current
    const target = timeline[selectedTimestep]
    if (!video || !target) {
      return
    }
    if (Math.abs(video.currentTime - target.start_s) > 0.35) {
      syncGuardRef.current = performance.now() + 280
      video.currentTime = target.start_s
      playbackStateRef.current.time = target.start_s
    }
  }, [selectedTimestep, timeline])

  const handleVideoTimeUpdate = () => {
    if (!timeline.length || performance.now() < syncGuardRef.current) {
      return
    }
    const video = videoRef.current
    if (!video) {
      return
    }
    playbackStateRef.current.time = video.currentTime
    const nextIndex = findTimestepIndex(timeline, video.currentTime)
    if (nextIndex !== selectedTimestep) {
      startTransition(() => {
        setSelectedTimestep(nextIndex)
      })
    }
  }

  async function handleUploadSubmit(event) {
    event.preventDefault()
    if (!uploadFile) {
      setError('Select a video first.')
      return
    }
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const job = await createRun(uploadFile, uploadSettings)
      setActiveJob(job)
      setNotice(`Processing ${uploadFile.name}`)
    } catch (caughtError) {
      setError(caughtError.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleAnalyze(activeId) {
    if (!activeId || !chatPrompt.trim()) {
      return
    }
    const userMessage = { role: 'user', content: chatPrompt.trim() }
    setChatMessages((current) => [...current, userMessage])
    setChatPrompt('')
    setChatBusy(true)
    setError('')
    try {
      const result = await analyzeRun(activeId, {
        ...analysisSettings,
        prompt: userMessage.content,
        previous_response_id: responseId,
      })
      setChatMessages((current) => [
        ...current,
        { role: 'assistant', content: result.reply },
      ])
      setResponseId(result.response_id ?? null)
    } catch (caughtError) {
      setError(caughtError.message)
    } finally {
      setChatBusy(false)
    }
  }

  return (
    <div className="shell">
      <div className="backdrop" />
      <header className="hero">
        <div className="hero__copy">
          <p className="hero__kicker">TRIBE v2 cortical inference</p>
          <div className="hero__brand">
            <img
              src="/branding/Veydalabs_logo+branding.png"
              alt="Veyda Labs"
              className="hero__logo"
            />
            <h1>Reel2Brain</h1>
          </div>
          <p className="hero__subtitle">
            Process video with Meta's TRIBE v2 model, then review synchronized cortical
            predictions across playback, static surface panels, zone curves, and an orbitable 3D
            cortex.
          </p>
        </div>
        <div className="hero__status">
          <span className={`status-pill ${health?.status === 'ok' ? 'status-pill--ok' : ''}`}>
            API {health?.status ?? 'starting'}
          </span>
          {activeJob ? (
            <span className="status-pill status-pill--live">
              {activeJob.progress_label} · {activeJob.progress_pct}%
            </span>
          ) : null}
        </div>
      </header>

      {error ? <div className="flash flash--error">{error}</div> : null}
      {notice ? <div className="flash flash--notice">{notice}</div> : null}

      <div className="layout">
        <aside className="sidebar">
          <section className="panel">
            <div className="panel__head">
              <div>
                <p className="panel__kicker">New inference</p>
                <h2>Process video</h2>
              </div>
            </div>
            <form className="upload-form" onSubmit={handleUploadSubmit}>
              <label className="file-drop">
                <input
                  type="file"
                  accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo"
                  onChange={(event) => handleUploadFileChange(event.target.files?.[0] ?? null)}
                />
                <span>{uploadFile ? uploadFile.name : 'Drop video file or browse'}</span>
              </label>

              {uploadPreviewUrl ? (
                <video src={uploadPreviewUrl} controls className="upload-preview" />
              ) : null}

              <div className="form-grid">
                <label>
                  TRIBE checkpoint
                  <input
                    value={uploadSettings.checkpoint}
                    onChange={(event) =>
                      setUploadSettings((current) => ({
                        ...current,
                        checkpoint: event.target.value,
                      }))
                    }
                  />
                </label>
                <label>
                  Compute device
                  <select
                    value={uploadSettings.device}
                    onChange={(event) =>
                      setUploadSettings((current) => ({
                        ...current,
                        device: event.target.value,
                      }))
                    }
                  >
                    <option value="cuda">cuda</option>
                    <option value="auto">auto</option>
                    <option value="cpu">cpu</option>
                  </select>
                </label>
                <label>
                  Data workers
                  <input
                    type="number"
                    min="0"
                    max="16"
                    value={uploadSettings.num_workers}
                    onChange={(event) =>
                      setUploadSettings((current) => ({
                        ...current,
                        num_workers: Number(event.target.value || 0),
                      }))
                    }
                  />
                </label>
              </div>

              <details className="disclosure">
                <summary>Language timing pipeline</summary>
                <div className="form-grid">
                  <label>
                    Text encoder
                    <input
                      value={uploadSettings.text_model_name}
                      onChange={(event) =>
                        setUploadSettings((current) => ({
                          ...current,
                          text_model_name: event.target.value,
                        }))
                      }
                    />
                  </label>
                  <label>
                    Alignment mode
                    <select
                      value={uploadSettings.text_mode}
                      onChange={(event) =>
                        setUploadSettings((current) => ({
                          ...current,
                          text_mode: event.target.value,
                        }))
                      }
                    >
                      <option value="paper">Paper-style timing</option>
                      <option value="direct">Direct transcript timing</option>
                    </select>
                  </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={uploadSettings.transcribe}
                      onChange={(event) =>
                        setUploadSettings((current) => ({
                          ...current,
                          transcribe: event.target.checked,
                        }))
                      }
                  />
                  <span>Transcribe video audio</span>
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={uploadSettings.staggered_sampling}
                    onChange={(event) =>
                      setUploadSettings((current) => ({
                        ...current,
                        staggered_sampling: event.target.checked,
                      }))
                    }
                  />
                  <span>Dual-pass 0.5s stagger (experimental)</span>
                </label>
                <p className="form-note">
                  Runs TRIBE twice and interleaves a second pass offset by <code>0.5s</code>.
                  Smoother temporal sampling, but not native higher-rate brain prediction.
                </p>
                <label>
                  Seconds / word
                  <input
                    type="number"
                      step="0.05"
                      min="0.1"
                      max="1.0"
                      value={uploadSettings.seconds_per_word}
                      onChange={(event) =>
                        setUploadSettings((current) => ({
                          ...current,
                          seconds_per_word: Number(event.target.value || 0.45),
                        }))
                      }
                    />
                  </label>
                  <label>
                    Context words
                    <input
                      type="number"
                      min="16"
                      max="256"
                      step="16"
                      value={uploadSettings.max_context_words}
                      onChange={(event) =>
                        setUploadSettings((current) => ({
                          ...current,
                          max_context_words: Number(event.target.value || 128),
                        }))
                      }
                    />
                  </label>
                </div>
              </details>

              <button type="submit" className="button button--primary" disabled={busy}>
                {busy ? 'Queueing job...' : 'Start TRIBE run'}
              </button>
            </form>
          </section>

          <section className="panel">
            <div className="panel__head">
              <div>
                <p className="panel__kicker">Processing</p>
                <h2>Job queue</h2>
              </div>
            </div>
            <JobRail jobs={jobs} />
          </section>

          <section className="panel">
            <div className="panel__head">
              <div>
                <p className="panel__kicker">Library</p>
                <h2>Run archive</h2>
              </div>
            </div>
            <RunLibrary
              runs={runs}
              activeRunId={activeRunId}
              onSelect={(runId) => {
                setError('')
                setNotice('')
                setActiveRunId(runId)
              }}
            />
          </section>
        </aside>

        <main className="stage">
          {!runDetail ? (
            <section className="panel panel--empty">
              <p className="panel__kicker">Ready</p>
              <h2>{loadingRun ? 'Loading run...' : 'Select or create a run'}</h2>
              <p className="panel-note">
                Upload a video from the left rail or open a saved run to inspect synchronized
                cortical predictions.
              </p>
            </section>
          ) : (
            <>
              <section className="panel">
                <div className="panel__head">
                  <div>
                    <p className="panel__kicker">Run summary</p>
                    <h2>{runDetail.subtitle || runDetail.title}</h2>
                    {runDetail.run_metadata?.sampling_label ? (
                      <p className="panel-note">
                        Mode: <strong>{runDetail.run_metadata.sampling_label}</strong>.{' '}
                        {runDetail.run_metadata.sampling_note}
                      </p>
                    ) : null}
                  </div>
                </div>
                <div className="metric-strip">
                  <MetricCard label="Timesteps" value={runDetail.timesteps} accent="#6ee7f9" />
                  <MetricCard label="Surface vertices" value={runDetail.vertex_count} accent="#22c55e" />
                  <MetricCard label="Input events" value={runDetail.events_count} accent="#f97316" />
                  <MetricCard
                    label="Mean |signal|"
                    value={runDetail.mean_abs.toFixed(4)}
                    accent="#facc15"
                  />
                </div>
              </section>

              <section className="panel">
                <div className="panel__head">
                  <div>
                    <p className="panel__kicker">Playback sync</p>
                    <h2>Video and cortical panels</h2>
                  </div>
                  <div className="transport">
                    <span>
                      t{selectedTimestep} · {formatSeconds(timeline[selectedTimestep]?.start_s ?? 0)}
                    </span>
                  </div>
                </div>
                <div className="review-grid">
                  <div className="review-video">
                    {runDetail.source_url ? (
                      <video
                        ref={videoRef}
                        src={runDetail.source_url}
                        controls
                        loop
                        className="review-video__player"
                        onTimeUpdate={handleVideoTimeUpdate}
                        onPlay={() => {
                          playbackStateRef.current.isPlaying = true
                          setIsPlaying(true)
                        }}
                        onPause={() => {
                          playbackStateRef.current.isPlaying = false
                          playbackStateRef.current.time =
                            videoRef.current?.currentTime ?? playbackStateRef.current.time
                          setIsPlaying(false)
                        }}
                      />
                    ) : (
                      <div className="review-video__empty">Source video unavailable.</div>
                    )}
                    <TimelineScroller
                      timeline={timeline}
                      selectedTimestep={selectedTimestep}
                      onSelect={setSelectedTimestep}
                      isPlaying={isPlaying}
                      onTogglePlay={() => setIsPlaying((current) => !current)}
                    />
                  </div>
                  <div className="review-panel">
                    {activeFrame ? (
                      <img
                        src={activeFrame.panel_url}
                        alt={`Brain panel for timestep ${selectedTimestep}`}
                        className="brain-panel"
                      />
                    ) : (
                      <div className="brain-panel brain-panel--empty">Brain panel unavailable.</div>
                    )}
                    <div className="brain-panel__caption">
                      Left, right, and dorsal cortical views for the selected timestep.
                    </div>
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel__head">
                  <div>
                    <p className="panel__kicker">Interactive cortex</p>
                    <h2>3D cortical surface</h2>
                  </div>
                  <p className="panel-note">
                    Class mode: quantitative activation heatmap with optional atlas guides. Drag
                    to orbit. Right-drag or shift-drag to pan. Scroll to zoom.
                  </p>
                </div>
                <div className="brain-stage">
                  <div className="brain-stage__viewer">
                    <Suspense fallback={<div className="brain-stage__loading">Loading 3D renderer...</div>}>
                      <ThreeBrainViewer
                        mesh={mesh}
                        frames={brainFrames?.frames ?? []}
                        selectedTimestep={selectedTimestep}
                        playbackRef={playbackStateRef}
                        overlayMode={brainOverlayMode}
                        className="brain-stage__canvas"
                      />
                    </Suspense>
                    <TimelineScroller
                      timeline={timeline}
                      selectedTimestep={selectedTimestep}
                      onSelect={setSelectedTimestep}
                      isPlaying={isPlaying}
                      onTogglePlay={() => setIsPlaying((current) => !current)}
                      className="timeline-card--brain-linked"
                    />
                  </div>
                  <div className="brain-stage__meta">
                    <ActivationScaleCard displayMeta={runDetail.brain_display_meta} />
                    <BrainOverlayCard
                      overlayMode={brainOverlayMode}
                      onOverlayChange={setBrainOverlayMode}
                      mesh={mesh}
                      displayMeta={runDetail.brain_display_meta}
                    />
                    <div>
                      <span>Current timestep</span>
                      <strong>t{selectedTimestep}</strong>
                    </div>
                    <div>
                      <span>Video time</span>
                      <strong>{formatSeconds(activeFrame?.start_s ?? 0)}</strong>
                    </div>
                    <div>
                      <span>Mean |signal|</span>
                      <strong>{Number(activeFrame?.mean_abs ?? 0).toFixed(4)}</strong>
                    </div>
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel__head">
                  <div>
                    <p className="panel__kicker">Cortical dynamics</p>
                    <h2>Cortical zone dynamics</h2>
                  </div>
                </div>
                <ZoneChart
                  series={runDetail.zone_series ?? []}
                  selectedTimestep={selectedTimestep}
                  onSelect={(value) => setSelectedTimestep(value)}
                />
              </section>

              <section className="panel">
                <div className="panel__head">
                  <div>
                    <p className="panel__kicker">Cortical systems</p>
                    <h2>Dominant activation profile</h2>
                  </div>
                </div>
                <div className="zone-summary">
                  {(runDetail.dominant_zones ?? []).map((zone) => (
                    <article key={zone.zone_key ?? zone.zone} className="zone-summary__card">
                      <div
                        className="zone-summary__accent"
                        style={{ background: getZoneColor(zone.zone_key) }}
                      />
                      <div>
                        <strong>{zone.zone}</strong>
                        <span>{(Number(zone.share ?? 0) * 100).toFixed(1)}% of aggregate activity</span>
                        <small>{zone.systems}</small>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <ChatPanel
                activeRunId={activeRunId}
                analysisSettings={analysisSettings}
                onSettingsChange={setAnalysisSettings}
                messages={chatMessages}
                prompt={chatPrompt}
                onPromptChange={setChatPrompt}
                onSubmit={handleAnalyze}
                busy={chatBusy}
              />
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
