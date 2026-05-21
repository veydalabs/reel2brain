from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
import logging
import mimetypes
from pathlib import Path
import pickle
import typing as tp

import numpy as np
from PIL import Image

from tribev2.easy import ImageComparisonRun, MultiModalRun, PredictionRun, segment_preview


LOGGER = logging.getLogger("tribev2.run_store")


def guess_media_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        return "video/mp4"
    if path.suffix.lower() == ".webm":
        return "video/webm"
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    if path.suffix.lower() == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


def get_saved_runs_folder(cache_folder: Path) -> Path:
    return cache_folder / "saved_runs"


def _format_run_input_kind(run: PredictionRun | ImageComparisonRun) -> str:
    if isinstance(run, ImageComparisonRun):
        mapping = {
            "image": "Images",
            "video": "Videos",
            "audio": "Audio",
            "text": "Texts",
        }
        return f"{mapping.get(run.compare_kind, run.compare_kind.title())} x{len(run.runs)}"
    if isinstance(run, MultiModalRun):
        ordered: list[str] = []
        labels = {"video": "Video", "image": "Image", "audio": "Audio", "text": "Text"}
        for modality in ("video", "image", "audio", "text"):
            if modality in run.component_runs:
                ordered.append(labels[modality])
        return " + ".join(ordered) if ordered else "Multimodal"
    return {
        "video": "Video",
        "audio": "Audio",
        "text": "Text",
        "image": "Image",
    }.get(run.input_kind, run.input_kind.title())


def _fit_image_bytes_to_height(
    raw_bytes: bytes,
    *,
    max_height: int = 96,
) -> bytes:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if image.height > max_height:
        scale = max_height / image.height
        image = image.resize(
            (max(1, int(image.width * scale)), max_height),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_array_to_png_bytes(
    image_array: np.ndarray,
    *,
    max_height: int = 96,
) -> bytes:
    image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
    if image.height > max_height:
        scale = max_height / image.height
        image = image.resize(
            (max(1, int(image.width * scale)), max_height),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _truncate_saved_text(value: str, *, limit: int = 88) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _extract_visual_preview_bytes(run: PredictionRun | ImageComparisonRun) -> bytes | None:
    visual_run: PredictionRun | None = None
    if isinstance(run, ImageComparisonRun):
        visual_run = next(
            (item for item in run.runs if item.input_kind in {"image", "video"}),
            None,
        )
    elif isinstance(run, MultiModalRun):
        visual_run = run.component_runs.get("image") or run.component_runs.get("video")
    elif run.input_kind in {"image", "video"}:
        visual_run = run

    if visual_run is None:
        return None
    if (
        visual_run.input_kind == "image"
        and visual_run.source_path is not None
        and visual_run.source_path.exists()
    ):
        return _fit_image_bytes_to_height(visual_run.source_path.read_bytes())
    if visual_run.input_kind == "video" and visual_run.segments:
        try:
            preview = segment_preview(visual_run, 0)
            frame = preview.get("frame")
            if frame is not None:
                return _image_array_to_png_bytes(np.asarray(frame, dtype=np.uint8))
        except Exception:
            LOGGER.exception(
                "Failed to build saved run video preview | source=%s",
                visual_run.source_path,
            )
    return None


def _saved_run_card_text(run: PredictionRun | ImageComparisonRun) -> tuple[str, str]:
    kind_label = _format_run_input_kind(run)
    if isinstance(run, ImageComparisonRun):
        if run.compare_kind == "text":
            texts = [_truncate_saved_text(item.raw_text or "") for item in run.runs[:2]]
            return kind_label, " / ".join(text for text in texts if text) or kind_label
        if run.compare_kind in {"audio", "image", "video"}:
            names = [item.source_path.name for item in run.runs if item.source_path is not None]
            return kind_label, " / ".join(names) or kind_label
        return kind_label, kind_label
    if isinstance(run, MultiModalRun):
        if run.raw_text:
            return kind_label, _truncate_saved_text(run.raw_text)
        names = [path.name for _, path in sorted(run.source_paths.items())]
        return kind_label, " · ".join(names) or kind_label
    if run.input_kind == "text":
        return kind_label, _truncate_saved_text(run.raw_text or "")
    if run.source_path is not None:
        return kind_label, run.source_path.name
    return kind_label, kind_label


def get_run_cache_key(run: PredictionRun) -> str:
    source_key = "none"
    if run.source_path is not None and run.source_path.exists():
        stat = run.source_path.stat()
        source_key = f"{run.source_path}:{stat.st_size}:{stat.st_mtime_ns}"
    signal_key = (
        f"{run.input_kind}:{run.preds.shape}:"
        f"{float(np.abs(run.preds).mean()):.6f}:{float(np.abs(run.preds).max()):.6f}"
    )
    return f"{source_key}:{signal_key}"


def get_saved_run_key(run: PredictionRun | ImageComparisonRun) -> str:
    if isinstance(run, ImageComparisonRun):
        inner = "|".join(get_run_cache_key(item) for item in run.runs)
        return f"comparison:{run.compare_kind}:{inner}"
    if isinstance(run, MultiModalRun):
        parts = [get_run_cache_key(run)]
        parts.extend(
            f"{modality}:{get_run_cache_key(component)}"
            for modality, component in sorted(run.component_runs.items())
        )
        return "multimodal:" + "|".join(parts)
    return get_run_cache_key(run)


def get_saved_run_id(run: PredictionRun | ImageComparisonRun) -> str:
    run_key = get_saved_run_key(run)
    return hashlib.sha1(run_key.encode("utf-8")).hexdigest()[:16]


def _serialize_prediction_run(run: PredictionRun) -> dict[str, tp.Any]:
    payload: dict[str, tp.Any] = {
        "kind": "prediction",
        "events": run.events,
        "preds": run.preds,
        "segments": run.segments,
        "input_kind": run.input_kind,
        "source_path": str(run.source_path) if run.source_path is not None else None,
        "raw_text": run.raw_text,
    }
    if isinstance(run, MultiModalRun):
        payload["kind"] = "multimodal"
        payload["component_runs"] = {
            modality: _serialize_prediction_run(component)
            for modality, component in run.component_runs.items()
        }
        payload["source_paths"] = {
            modality: str(path) for modality, path in run.source_paths.items()
        }
        payload["primary_input_kind"] = run.primary_input_kind
    return payload


def _serialize_saved_run(run: PredictionRun | ImageComparisonRun) -> dict[str, tp.Any]:
    if isinstance(run, ImageComparisonRun):
        return {
            "kind": "comparison",
            "compare_kind": run.compare_kind,
            "runs": [_serialize_prediction_run(item) for item in run.runs],
        }
    return _serialize_prediction_run(run)


def _deserialize_prediction_run(payload: dict[str, tp.Any]) -> PredictionRun:
    source_path = payload.get("source_path")
    common_kwargs = {
        "events": payload["events"],
        "preds": payload["preds"],
        "segments": payload.get("segments", []),
        "input_kind": str(payload["input_kind"]),
        "source_path": Path(source_path) if source_path else None,
        "raw_text": payload.get("raw_text"),
    }
    if payload.get("kind") == "multimodal":
        return MultiModalRun(
            **common_kwargs,
            component_runs={
                modality: _deserialize_prediction_run(component_payload)
                for modality, component_payload in tp.cast(
                    dict[str, dict[str, tp.Any]],
                    payload.get("component_runs", {}),
                ).items()
            },
            source_paths={
                modality: Path(path)
                for modality, path in tp.cast(
                    dict[str, str],
                    payload.get("source_paths", {}),
                ).items()
            },
            primary_input_kind=tp.cast(str | None, payload.get("primary_input_kind")),
        )
    return PredictionRun(**common_kwargs)


def _deserialize_saved_run(payload: dict[str, tp.Any]) -> PredictionRun | ImageComparisonRun:
    if payload.get("kind") == "comparison":
        return ImageComparisonRun(
            runs=[
                _deserialize_prediction_run(item)
                for item in tp.cast(list[dict[str, tp.Any]], payload.get("runs", []))
            ],
            compare_kind=str(payload.get("compare_kind", "image")),
        )
    return _deserialize_prediction_run(payload)


def persist_saved_run(cache_folder: Path, run: PredictionRun | ImageComparisonRun) -> str:
    saved_root = get_saved_runs_folder(cache_folder)
    saved_root.mkdir(parents=True, exist_ok=True)
    run_key = get_saved_run_key(run)
    run_id = get_saved_run_id(run)
    run_folder = saved_root / run_id
    run_folder.mkdir(parents=True, exist_ok=True)

    title, subtitle = _saved_run_card_text(run)
    preview_bytes = _extract_visual_preview_bytes(run)
    preview_file = None
    if preview_bytes is not None:
        preview_path = run_folder / "preview.png"
        preview_path.write_bytes(preview_bytes)
        preview_file = preview_path.name

    metadata_path = run_folder / "metadata.json"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = str(previous.get("created_at", created_at))
        except Exception:
            LOGGER.exception(
                "Failed to read saved run metadata | path=%s",
                metadata_path,
            )

    metadata = {
        "id": run_id,
        "run_key": run_key,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind_label": title,
        "subtitle": subtitle,
        "preview_file": preview_file,
        "input_kind": run.compare_kind if isinstance(run, ImageComparisonRun) else run.input_kind,
        "is_comparison": isinstance(run, ImageComparisonRun),
        "is_multimodal": isinstance(run, MultiModalRun),
        "timesteps": len(run.runs[0].preds) if isinstance(run, ImageComparisonRun) and run.runs else len(run.preds),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "version": 2,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": _serialize_saved_run(run),
    }
    with open(run_folder / "run.pkl", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return run_id


def load_saved_run(cache_folder: Path, run_id: str) -> PredictionRun | ImageComparisonRun:
    run_path = get_saved_runs_folder(cache_folder) / run_id / "run.pkl"
    with open(run_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "run" in payload:
        return _deserialize_saved_run(tp.cast(dict[str, tp.Any], payload["run"]))
    return tp.cast(PredictionRun | ImageComparisonRun, payload)


def list_saved_runs(cache_folder: Path) -> list[dict[str, tp.Any]]:
    saved_root = get_saved_runs_folder(cache_folder)
    if not saved_root.exists():
        return []
    entries: list[dict[str, tp.Any]] = []
    for metadata_path in saved_root.glob("*/metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.exception(
                "Failed to parse saved run metadata | path=%s",
                metadata_path,
            )
            continue
        payload["folder"] = str(metadata_path.parent)
        entries.append(payload)
    entries.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return entries


__all__ = [
    "get_saved_run_id",
    "get_saved_run_key",
    "get_saved_runs_folder",
    "guess_media_mime",
    "list_saved_runs",
    "load_saved_run",
    "persist_saved_run",
]
