# Reel2Brain

Production web app for running Meta's TRIBE v2 model locally on video and reviewing the predicted cortical response in a synchronized scientific UI.

## What This Repo Is

This repository is the public `Reel2Brain` application surface.

- FastAPI backend for uploads, job orchestration, saved runs, and optional OpenAI analysis
- React frontend for playback review, run library, and timeline navigation
- Three.js cortex viewer with synchronized TRIBE timestep playback
- Static cortical panels and zone-dynamics charts generated from TRIBE predictions

The old Streamlit and dashboard surfaces have been removed from this repo so the public project stays focused on the production app.

## Requirements

- Python `3.11+`
- Node.js `20+`
- A working local TRIBE v2 environment with the required model dependencies

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[plotting,webapp]"
cd frontend
npm install
npm run build
cd ..
```

## Run

```bash
source .venv/bin/activate
reel2brain
```

The app serves on `http://localhost:8000`.

## Frontend Development

Use the FastAPI backend for inference and API routes, then run Vite separately for frontend iteration:

```bash
source .venv/bin/activate
reel2brain
```

In a second shell:

```bash
cd frontend
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies `/api` to the backend on port `8000`.

## Current Product Scope

- Upload a video and queue a TRIBE v2 run
- Review source playback and cortical outputs on a shared timeline
- Inspect left, right, and dorsal cortical panels per timestep
- Orbit and pan a larger 3D cortical surface
- Browse saved runs in a persistent local library
- Optionally send run context and selected cortical panels to OpenAI for interpretation

## Repo Layout

```text
frontend/              React + Vite + Three.js client
public/branding/       Veyda Labs branding assets used by the app
tribev2/production_api.py
tribev2/run_store.py   Saved-run persistence and preview generation
tribev2/easy.py        TRIBE inference and rendering helpers
tribev2/openai_chat.py Optional OpenAI analysis pipeline
```

## Attribution

Reel2Brain is built on top of Meta's TRIBE v2 model and related research code.

- Paper: [A foundation model of vision, audition, and language for in silico neuroscience](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- Original repository: [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)

If you publish work derived from this app or its outputs, cite the original TRIBE v2 work.
