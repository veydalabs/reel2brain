async function readJson(response) {
  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? detail
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(detail || `Request failed with status ${response.status}`)
  }
  return response.json()
}

export async function fetchHealth() {
  return readJson(await fetch('/api/health'))
}

export async function fetchRuns() {
  return readJson(await fetch('/api/runs'))
}

export async function fetchJobs() {
  return readJson(await fetch('/api/jobs'))
}

export async function fetchJob(jobId) {
  return readJson(await fetch(`/api/jobs/${jobId}`))
}

export async function createRun(file, settings) {
  const form = new FormData()
  form.set('video', file)
  Object.entries(settings).forEach(([key, value]) => {
    form.set(key, String(value))
  })
  return readJson(
    await fetch('/api/runs', {
      method: 'POST',
      body: form,
    }),
  )
}

export async function fetchRunDetail(runId) {
  return readJson(await fetch(`/api/runs/${runId}`))
}

export async function fetchBrainMesh() {
  return readJson(await fetch('/api/brain/mesh'))
}

export async function fetchBrainFrames(runId) {
  return readJson(await fetch(`/api/runs/${runId}/brain/frames`))
}

export async function analyzeRun(runId, payload) {
  return readJson(
    await fetch(`/api/runs/${runId}/analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    }),
  )
}
