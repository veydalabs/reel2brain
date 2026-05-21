from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tribev2.easy import PredictionRun
from tribev2.run_store import (
    get_saved_runs_folder,
    guess_media_mime,
    list_saved_runs,
    load_saved_run,
    persist_saved_run,
)


def _build_text_run(tmp_path: Path) -> PredictionRun:
    return PredictionRun(
        events=pd.DataFrame(
            [
                {
                    "type": "Word",
                    "start": 0.0,
                    "duration": 1.0,
                    "text": "hello world",
                }
            ]
        ),
        preds=np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float),
        segments=[],
        input_kind="text",
        source_path=tmp_path / "script.txt",
        raw_text="hello world",
    )


def test_guess_media_mime_prefers_known_video_suffixes() -> None:
    assert guess_media_mime(Path("clip.mp4")) == "video/mp4"
    assert guess_media_mime(Path("clip.webm")) == "video/webm"


def test_persist_saved_run_roundtrip_for_text(tmp_path: Path) -> None:
    run = _build_text_run(tmp_path)

    run_id = persist_saved_run(tmp_path, run)
    entries = list_saved_runs(tmp_path)
    loaded = load_saved_run(tmp_path, run_id)

    assert entries[0]["id"] == run_id
    assert entries[0]["preview_file"] is None
    assert loaded.input_kind == "text"
    assert loaded.raw_text == "hello world"
    np.testing.assert_allclose(loaded.preds, run.preds)


def test_persist_saved_run_generates_image_preview(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (240, 180), (25, 180, 120)).save(image_path)

    run = PredictionRun(
        events=pd.DataFrame([{"type": "Image", "start": 0.0, "duration": 1.0}]),
        preds=np.array([[0.1, 0.2]], dtype=float),
        segments=[],
        input_kind="image",
        source_path=image_path,
        raw_text=None,
    )

    run_id = persist_saved_run(tmp_path, run)
    entries = list_saved_runs(tmp_path)
    preview_path = get_saved_runs_folder(tmp_path) / run_id / "preview.png"

    assert entries[0]["id"] == run_id
    assert entries[0]["preview_file"] == "preview.png"
    assert preview_path.exists()
