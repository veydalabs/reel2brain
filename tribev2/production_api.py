from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import html
import json
import logging
from pathlib import Path
import re
import threading
import traceback
import typing as tp
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from tribev2.run_store import (
    get_saved_runs_folder,
    guess_media_mime,
    list_saved_runs,
    load_saved_run,
    persist_saved_run,
)
from tribev2.easy import (
    DEFAULT_TEXT_MODEL,
    ImageComparisonRun,
    PredictionRun,
    _smooth_surface_values,
    _get_surface_render_data,
    ZONE_FAMILY_META,
    build_display_reference_signal,
    build_emotion_hypothesis_frame,
    build_run_roi_frame,
    build_run_zone_frame,
    build_timestep_report_frame,
    build_timestep_zone_frame,
    classify_roi_family,
    collect_timestep_metadata,
    get_pyvista_plotter,
    load_model,
    normalize_signal_for_display,
    predict_from_prepared_events,
    prepare_events,
    render_run_panel_bytes,
    summarize_predictions,
)
from tribev2.openai_chat import DEFAULT_OPENAI_CHAT_MODEL, request_openai_run_explanation
from tribev2.runtime import apply_warning_filters
from tribev2.utils import get_hcp_vertex_labels


LOGGER = logging.getLogger("tribev2.production_api")
DEFAULT_CACHE = Path("./cache")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
JOB_LOCK = threading.Lock()


class AnalysisRequest(BaseModel):
    prompt: str
    api_key: str
    model: str = DEFAULT_OPENAI_CHAT_MODEL
    reasoning_effort: str = "medium"
    image_detail: str = "low"
    max_images: int = 4
    previous_response_id: str | None = None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sanitize_filename(name: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("_", name).strip("._")
    return cleaned or "upload.bin"


def to_bool(value: str | bool | None, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def frame_to_records(frame) -> list[dict[str, tp.Any]]:
    if frame is None or frame.empty:
        return []
    return tp.cast(list[dict[str, tp.Any]], json.loads(frame.to_json(orient="records")))


def get_jobs_folder(cache_folder: Path) -> Path:
    return cache_folder / "web_jobs"


def get_job_folder(cache_folder: Path, job_id: str) -> Path:
    return get_jobs_folder(cache_folder) / job_id


def get_job_meta_path(cache_folder: Path, job_id: str) -> Path:
    return get_job_folder(cache_folder, job_id) / "job.json"


def read_job(cache_folder: Path, job_id: str) -> dict[str, tp.Any]:
    path = get_job_meta_path(cache_folder, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")
    return tp.cast(dict[str, tp.Any], json.loads(path.read_text(encoding="utf-8")))


def write_job(cache_folder: Path, payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
    job_id = str(payload["id"])
    folder = get_job_folder(cache_folder, job_id)
    folder.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now_iso()
    get_job_meta_path(cache_folder, job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def update_job(cache_folder: Path, job_id: str, **changes: tp.Any) -> dict[str, tp.Any]:
    with JOB_LOCK:
        payload = read_job(cache_folder, job_id)
        payload.update(changes)
        return write_job(cache_folder, payload)


def list_jobs(cache_folder: Path) -> list[dict[str, tp.Any]]:
    root = get_jobs_folder(cache_folder)
    if not root.exists():
        return []
    jobs: list[dict[str, tp.Any]] = []
    for path in root.glob("*/job.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.exception("Failed to parse job metadata | path=%s", path)
            continue
        jobs.append(tp.cast(dict[str, tp.Any], payload))
    jobs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return jobs


def mark_interrupted_jobs(cache_folder: Path) -> None:
    for job in list_jobs(cache_folder):
        if job.get("status") in {"queued", "processing"}:
            update_job(
                cache_folder,
                str(job["id"]),
                status="failed",
                error="The server restarted before this job completed.",
                progress_pct=0,
                progress_label="Interrupted",
            )


def create_job(
    cache_folder: Path,
    *,
    upload_name: str,
    source_path: Path,
    options: dict[str, tp.Any],
) -> dict[str, tp.Any]:
    job_id = uuid.uuid4().hex[:16]
    payload = {
        "id": job_id,
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "source_name": upload_name,
        "source_path": str(source_path),
        "options": options,
        "progress_pct": 0,
        "progress_label": "Queued",
        "saved_run_id": None,
        "error": None,
    }
    with JOB_LOCK:
        return write_job(cache_folder, payload)


def get_saved_run_metadata_path(cache_folder: Path, run_id: str) -> Path:
    return get_saved_runs_folder(cache_folder) / run_id / "metadata.json"


def read_saved_run_metadata(cache_folder: Path, run_id: str) -> dict[str, tp.Any]:
    path = get_saved_run_metadata_path(cache_folder, run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return tp.cast(dict[str, tp.Any], json.loads(path.read_text(encoding="utf-8")))


def get_saved_run_artifacts_folder(cache_folder: Path, run_id: str) -> Path:
    folder = get_saved_runs_folder(cache_folder) / run_id / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_saved_run_artifact_path(cache_folder: Path, run_id: str, filename: str) -> Path:
    return get_saved_run_artifacts_folder(cache_folder, run_id) / filename


def load_prediction_run(cache_folder: Path, run_id: str) -> PredictionRun:
    run = load_saved_run(cache_folder, run_id)
    if isinstance(run, ImageComparisonRun):
        raise HTTPException(
            status_code=400,
            detail="This production API currently serves single-run predictions only.",
        )
    return tp.cast(PredictionRun, run)


def build_zone_series_payload(run: PredictionRun) -> list[dict[str, tp.Any]]:
    run_zone_frame = build_run_zone_frame(run)
    timestep_zone_frame = build_timestep_zone_frame(run)
    if run_zone_frame.empty or timestep_zone_frame.empty:
        return []
    pivot = (
        timestep_zone_frame.pivot(index="timestep", columns="zone_key", values="share")
        .fillna(0.0)
        .sort_index()
    )
    ordered_columns = [
        zone_key for zone_key in run_zone_frame["zone_key"].tolist() if zone_key in pivot.columns
    ]
    if ordered_columns:
        pivot = pivot.reindex(columns=ordered_columns)
    labels_by_key = {
        str(row["zone_key"]): str(row["zone"])
        for row in run_zone_frame[["zone_key", "zone"]].to_dict(orient="records")
    }
    series: list[dict[str, tp.Any]] = []
    for zone_key in pivot.columns:
        values = [round(float(value), 6) for value in pivot[zone_key].astype(float).tolist()]
        series.append(
            {
                "zone_key": str(zone_key),
                "zone": labels_by_key.get(str(zone_key), str(zone_key)),
                "values": values,
                "peak": round(max(values) if values else 0.0, 6),
            }
        )
    return series


def build_timeline_payload(run: PredictionRun, run_id: str) -> list[dict[str, tp.Any]]:
    timing = collect_timestep_metadata(run)
    summary = summarize_predictions(run.preds)
    rows: list[dict[str, tp.Any]] = []
    for timestep in range(len(run.preds)):
        meta = timing[timestep] if timestep < len(timing) else {}
        summary_row = summary.iloc[timestep] if timestep < len(summary) else None
        rows.append(
            {
                "timestep": timestep,
                "start_s": round(float(meta.get("start", timestep)), 3),
                "duration_s": round(float(meta.get("duration", 1.0)), 3),
                "text": str(meta.get("text", "") or ""),
                "mean_abs": round(float(summary_row["mean_abs"]), 6) if summary_row is not None else 0.0,
                "max_abs": round(float(summary_row["max_abs"]), 6) if summary_row is not None else 0.0,
                "panel_url": f"/api/runs/{run_id}/panels/{timestep}.jpg",
            }
        )
    return rows


def build_run_entry(entry: dict[str, tp.Any]) -> dict[str, tp.Any]:
    run_id = str(entry["id"])
    preview_url = f"/api/runs/{run_id}/preview" if entry.get("preview_file") else None
    return {
        "id": run_id,
        "title": str(entry.get("kind_label", "Run")),
        "subtitle": str(entry.get("subtitle", "")),
        "updated_at": str(entry.get("updated_at", "")),
        "created_at": str(entry.get("created_at", "")),
        "timesteps": int(entry.get("timesteps", 0) or 0),
        "input_kind": str(entry.get("input_kind", "")),
        "preview_url": preview_url,
        "detail_url": f"/api/runs/{run_id}",
    }


@lru_cache(maxsize=4)
def get_mesh_payload(mesh: str = "fsaverage5") -> dict[str, tp.Any]:
    plotter = get_pyvista_plotter(mesh)
    mesh_data = plotter._mesh["both"]
    coords = np.round(np.asarray(mesh_data["coords"], dtype=float), 3).tolist()
    faces = np.asarray(mesh_data["faces"], dtype=int).tolist()
    bg_map = np.asarray(mesh_data["bg_map"], dtype=float)
    bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
    bg_bytes = np.round(np.clip(bg_norm, 0.0, 1.0) * 255).astype(np.uint8)

    zone_keys = list(ZONE_FAMILY_META.keys())
    zone_index_by_key = {zone_key: index for index, zone_key in enumerate(zone_keys)}
    vertex_zone_labels = get_hcp_vertex_labels(mesh=mesh, combine=False)
    parcel_label_to_index: dict[str, int] = {}
    parcel_indices: list[int] = []
    for vertex_label in vertex_zone_labels:
        label = vertex_label or "Unknown"
        if label not in parcel_label_to_index:
            parcel_label_to_index[label] = len(parcel_label_to_index)
        parcel_indices.append(parcel_label_to_index[label])
    zone_indices = [
        zone_index_by_key.get(
            classify_roi_family(vertex_label) if vertex_label else "association_other",
            zone_index_by_key["association_other"],
        )
        for vertex_label in vertex_zone_labels
    ]

    return {
        "mesh": mesh,
        "coords": coords,
        "faces": faces,
        "bg_b64": base64.b64encode(bg_bytes.tobytes()).decode("ascii"),
        "zone_keys": zone_keys,
        "zone_labels": {
            zone_key: str(zone_meta["label"])
            for zone_key, zone_meta in ZONE_FAMILY_META.items()
        },
        "zone_indices": zone_indices,
        "parcel_indices": parcel_indices,
        "parcel_count": len(parcel_label_to_index),
    }


@lru_cache(maxsize=4)
def get_cached_model(
    checkpoint: str,
    cache_folder: str,
    device: str,
    num_workers: int,
    text_model_name: str,
):
    return load_model(
        checkpoint=checkpoint,
        cache_folder=cache_folder,
        device=device,
        num_workers=num_workers,
        text_model_name=text_model_name,
    )


def get_cached_brain_frames_payload(
    cache_folder: Path,
    run_id: str,
    run: PredictionRun,
    *,
    mesh: str = "fsaverage5",
) -> dict[str, tp.Any]:
    artifact_path = get_saved_run_artifact_path(cache_folder, run_id, "brain_frames.json")
    if artifact_path.exists():
        cached_payload = tp.cast(dict[str, tp.Any], json.loads(artifact_path.read_text(encoding="utf-8")))
        frames = tp.cast(list[dict[str, tp.Any]], cached_payload.get("frames", []))
        if frames and all("activation_b64" in frame for frame in frames):
            return cached_payload

    display_reference = build_display_reference_signal(run)
    timing = collect_timestep_metadata(run)
    summary = summarize_predictions(run.preds)
    plotter = get_pyvista_plotter(mesh)
    mesh_data = plotter._mesh["both"]
    frames: list[dict[str, tp.Any]] = []
    for timestep in range(len(run.preds)):
        meta = timing[timestep] if timestep < len(timing) else {}
        summary_row = summary.iloc[timestep] if timestep < len(summary) else None
        normalized_signal = normalize_signal_for_display(
            run.preds[timestep],
            percentile=99,
            reference_signal=display_reference,
        )
        activation = plotter.get_stat_map(normalized_signal)["both"]
        activation = _smooth_surface_values(
            activation,
            mesh=mesh,
            passes=2,
            blend=0.34,
        )
        activation = np.clip(activation, 0.0, 1.0)
        _, _, _, colors = _get_surface_render_data(
            run.preds[timestep],
            mesh=mesh,
            cmap="fire",
            norm_percentile=99,
            vmin=0.5,
            display_reference=display_reference,
            surface_smoothing_passes=2,
            surface_smoothing_blend=0.34,
        )
        color_bytes = np.round(colors * 255).astype(np.uint8).tobytes()
        activation_bytes = np.round(activation * 255).astype(np.uint8).tobytes()
        frames.append(
            {
                "timestep": timestep,
                "start_s": round(float(meta.get("start", timestep)), 3),
                "duration_s": round(float(meta.get("duration", 1.0)), 3),
                "text": str(meta.get("text", "") or ""),
                "mean_abs": round(float(summary_row["mean_abs"]), 6) if summary_row is not None else 0.0,
                "colors_b64": base64.b64encode(color_bytes).decode("ascii"),
                "activation_b64": base64.b64encode(activation_bytes).decode("ascii"),
                "panel_url": f"/api/runs/{run_id}/panels/{timestep}.jpg",
            }
        )

    payload = {
        "mesh": mesh,
        "vertex_count": int(mesh_data["coords"].shape[0]),
        "frames": frames,
    }
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def get_cached_panel_bytes(
    cache_folder: Path,
    run_id: str,
    run: PredictionRun,
    timestep: int,
) -> bytes:
    artifact_path = get_saved_run_artifact_path(cache_folder, run_id, f"panel_{timestep:04d}.jpg")
    if artifact_path.exists():
        return artifact_path.read_bytes()
    image_bytes = render_run_panel_bytes(
        run,
        timestep=timestep,
        image_format="JPEG",
        quality=86,
        display_reference=build_display_reference_signal(run),
    )
    artifact_path.write_bytes(image_bytes)
    return image_bytes


def build_run_detail_payload(cache_folder: Path, run_id: str, run: PredictionRun) -> dict[str, tp.Any]:
    metadata = read_saved_run_metadata(cache_folder, run_id)
    source_url = None
    source_mime = None
    source_name = None
    if run.source_path is not None and run.source_path.exists():
        source_url = f"/api/runs/{run_id}/source"
        source_mime = guess_media_mime(run.source_path)
        source_name = run.source_path.name

    run_zone_frame = build_run_zone_frame(run)
    stats_frame = summarize_predictions(run.preds)
    mean_abs = float(stats_frame["mean_abs"].mean()) if not stats_frame.empty else 0.0
    peak_abs = float(stats_frame["max_abs"].max()) if not stats_frame.empty else 0.0
    return {
        "id": run_id,
        "title": str(metadata.get("kind_label", "Run")),
        "subtitle": str(metadata.get("subtitle", "")),
        "created_at": str(metadata.get("created_at", "")),
        "updated_at": str(metadata.get("updated_at", "")),
        "input_kind": run.input_kind,
        "timesteps": len(run.preds),
        "vertex_count": int(run.preds.shape[1]),
        "events_count": len(run.events),
        "mean_abs": round(mean_abs, 6),
        "peak_abs": round(peak_abs, 6),
        "source_name": source_name,
        "source_url": source_url,
        "source_mime": source_mime,
        "preview_url": f"/api/runs/{run_id}/preview" if metadata.get("preview_file") else None,
        "brain_mesh_url": "/api/brain/mesh",
        "brain_frames_url": f"/api/runs/{run_id}/brain/frames",
        "brain_display_meta": {
            "mesh": "fsaverage5",
            "activation_range": [0.0, 1.0],
            "activation_units": "normalized activation",
            "normalization": "shared_percentile_99_reference",
            "overlay_note": "Optional overlays are categorical guides and do not change activation values.",
        },
        "timeline": build_timeline_payload(run, run_id),
        "zone_series": build_zone_series_payload(run),
        "dominant_zones": frame_to_records(run_zone_frame.head(8)),
        "top_rois": frame_to_records(build_run_roi_frame(run, top_k=24)),
        "emotion_hypotheses": frame_to_records(build_emotion_hypothesis_frame(run)),
        "timestep_table": frame_to_records(build_timestep_report_frame(run)),
        "openai_defaults": {
            "model": DEFAULT_OPENAI_CHAT_MODEL,
            "reasoning_effort": "medium",
            "image_detail": "low",
            "max_images": 4,
        },
        "text_pipeline_defaults": {
            "text_model": DEFAULT_TEXT_MODEL,
            "text_mode": "paper",
            "transcribe": False,
            "seconds_per_word": 0.45,
            "max_context_words": 128,
        },
    }


def run_prediction_job(cache_folder: Path, job_id: str) -> None:
    try:
        update_job(cache_folder, job_id, status="processing", progress_pct=12, progress_label="Loading model")
        job = read_job(cache_folder, job_id)
        options = tp.cast(dict[str, tp.Any], job.get("options", {}))
        source_path = Path(str(job["source_path"]))
        model = get_cached_model(
            str(options["checkpoint"]),
            str(cache_folder),
            str(options["device"]),
            int(options["num_workers"]),
            str(options["text_model_name"]),
        )
        update_job(cache_folder, job_id, progress_pct=34, progress_label="Preparing events")
        events, input_kind = prepare_events(
            cache_folder=cache_folder,
            video_path=source_path,
            transcribe=bool(options["transcribe"]),
            direct_text=bool(options["direct_text"]),
            seconds_per_word=float(options["seconds_per_word"]),
            max_context_words=int(options["max_context_words"]),
        )
        update_job(cache_folder, job_id, progress_pct=68, progress_label="Running TRIBE")
        run = predict_from_prepared_events(
            model,
            events,
            input_kind=input_kind,
            source_path=source_path,
            verbose=False,
        )
        update_job(cache_folder, job_id, progress_pct=86, progress_label="Saving run")
        run_id = persist_saved_run(cache_folder, run)
        update_job(
            cache_folder,
            job_id,
            status="completed",
            progress_pct=100,
            progress_label="Complete",
            saved_run_id=run_id,
        )
    except Exception as exc:
        LOGGER.exception("Prediction job failed | job_id=%s", job_id)
        update_job(
            cache_folder,
            job_id,
            status="failed",
            progress_pct=0,
            progress_label="Failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def frontend_placeholder(frontend_dist: Path) -> HTMLResponse:
    message = f"""
    <html>
      <head>
        <title>Reel2Brain</title>
        <style>
          body {{
            margin: 0;
            font-family: "IBM Plex Sans", system-ui, sans-serif;
            background: radial-gradient(circle at top, #11263a, #060a10 48%);
            color: #e7eef5;
            min-height: 100vh;
            display: grid;
            place-items: center;
          }}
          main {{
            width: min(760px, 92vw);
            border: 1px solid rgba(103, 232, 249, 0.18);
            background: rgba(8, 13, 20, 0.92);
            padding: 28px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
          }}
          code {{
            background: rgba(148, 163, 184, 0.12);
            padding: 2px 6px;
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Reel2Brain API is running</h1>
          <p>The FastAPI backend is live, but the React build has not been generated yet.</p>
          <p>Expected frontend bundle directory:</p>
          <p><code>{html.escape(str(frontend_dist))}</code></p>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(message)


def create_app() -> FastAPI:
    cache_folder = DEFAULT_CACHE
    frontend_dist = Path(__file__).with_name("web_dist")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reel2brain")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        apply_warning_filters()
        cache_folder.mkdir(parents=True, exist_ok=True)
        mark_interrupted_jobs(cache_folder)
        app.state.cache_folder = cache_folder
        app.state.frontend_dist = frontend_dist
        app.state.executor = executor
        yield
        executor.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(
        title="Reel2Brain API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": "Reel2Brain"}

    @app.get("/api/jobs")
    async def get_jobs(request: Request) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        return {"items": list_jobs(cache)}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        return read_job(cache, job_id)

    @app.post("/api/runs")
    async def create_run(
        request: Request,
        video: UploadFile = File(...),
        checkpoint: str = Form("facebook/tribev2"),
        device: str = Form("cuda"),
        num_workers: int = Form(0),
        text_model_name: str = Form(DEFAULT_TEXT_MODEL),
        text_mode: str = Form("paper"),
        transcribe: str = Form("false"),
        seconds_per_word: float = Form(0.45),
        max_context_words: int = Form(128),
    ) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        file_name = sanitize_filename(video.filename or "video.mp4")
        suffix = Path(file_name).suffix or ".mp4"
        source_id = uuid.uuid4().hex[:12]
        source_dir = cache / "uploads" / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"source{suffix}"
        source_path.write_bytes(await video.read())

        options = {
            "checkpoint": checkpoint,
            "device": device,
            "num_workers": int(num_workers),
            "text_model_name": text_model_name,
            "text_mode": text_mode,
            "direct_text": text_mode == "direct",
            "transcribe": to_bool(transcribe),
            "seconds_per_word": float(seconds_per_word),
            "max_context_words": int(max_context_words),
        }
        job = create_job(
            cache,
            upload_name=file_name,
            source_path=source_path,
            options=options,
        )
        executor_ref = tp.cast(ThreadPoolExecutor, request.app.state.executor)
        executor_ref.submit(run_prediction_job, cache, str(job["id"]))
        return job

    @app.get("/api/runs")
    async def get_runs(request: Request) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        entries = [build_run_entry(entry) for entry in list_saved_runs(cache)]
        return {"items": entries}

    @app.get("/api/runs/{run_id}")
    async def get_run_detail(run_id: str, request: Request) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        run = load_prediction_run(cache, run_id)
        return build_run_detail_payload(cache, run_id, run)

    @app.get("/api/runs/{run_id}/preview")
    async def get_run_preview(run_id: str, request: Request):
        cache = tp.cast(Path, request.app.state.cache_folder)
        metadata = read_saved_run_metadata(cache, run_id)
        preview_file = str(metadata.get("preview_file") or "").strip()
        if not preview_file:
            raise HTTPException(status_code=404, detail="Run preview unavailable.")
        preview_path = get_saved_runs_folder(cache) / run_id / preview_file
        if not preview_path.exists():
            raise HTTPException(status_code=404, detail="Run preview file missing.")
        return FileResponse(preview_path, media_type="image/png")

    @app.get("/api/runs/{run_id}/source")
    async def get_run_source(run_id: str, request: Request):
        cache = tp.cast(Path, request.app.state.cache_folder)
        run = load_prediction_run(cache, run_id)
        if run.source_path is None or not run.source_path.exists():
            raise HTTPException(status_code=404, detail="Source file unavailable.")
        return FileResponse(run.source_path, media_type=guess_media_mime(run.source_path))

    @app.get("/api/runs/{run_id}/panels/{timestep}.jpg")
    async def get_run_panel(run_id: str, timestep: int, request: Request):
        cache = tp.cast(Path, request.app.state.cache_folder)
        run = load_prediction_run(cache, run_id)
        if timestep < 0 or timestep >= len(run.preds):
            raise HTTPException(status_code=404, detail="Invalid timestep.")
        return Response(
            content=get_cached_panel_bytes(cache, run_id, run, timestep),
            media_type="image/jpeg",
        )

    @app.get("/api/brain/mesh")
    async def get_brain_mesh(mesh: str = "fsaverage5") -> dict[str, tp.Any]:
        return get_mesh_payload(mesh)

    @app.get("/api/runs/{run_id}/brain/frames")
    async def get_brain_frames(run_id: str, request: Request, mesh: str = "fsaverage5") -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        run = load_prediction_run(cache, run_id)
        payload = get_cached_brain_frames_payload(cache, run_id, run, mesh=mesh)
        return JSONResponse(
            content=payload,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.post("/api/runs/{run_id}/analysis")
    async def analyze_run(run_id: str, body: AnalysisRequest, request: Request) -> dict[str, tp.Any]:
        cache = tp.cast(Path, request.app.state.cache_folder)
        run = load_prediction_run(cache, run_id)
        if not body.api_key.strip():
            raise HTTPException(status_code=400, detail="Missing OpenAI API key.")
        reply, response_id, labels = request_openai_run_explanation(
            api_key=body.api_key.strip(),
            model=body.model.strip(),
            reasoning_effort=body.reasoning_effort,
            user_prompt=body.prompt,
            run=run,
            previous_response_id=body.previous_response_id,
            include_context=body.previous_response_id is None,
            image_detail=body.image_detail,
            max_images=int(body.max_images),
        )
        return {"reply": reply, "response_id": response_id, "labels": labels}

    @app.get("/", include_in_schema=False)
    async def frontend_root():
        if not frontend_dist.exists():
            return frontend_placeholder(frontend_dist)
        return FileResponse(frontend_dist / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_catchall(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if not frontend_dist.exists():
            return frontend_placeholder(frontend_dist)
        requested = frontend_dist / full_path
        if requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
