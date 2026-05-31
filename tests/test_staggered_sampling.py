from pathlib import Path

import numpy as np
import pandas as pd

from neuralset.events.etypes import Word
from tribev2.easy import PredictionRun, StoredSegment, build_offset_events, build_staggered_prediction_run


def test_build_offset_events_shifts_and_clips_media_timeline() -> None:
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": "clip.mp4",
                "start": 0.0,
                "duration": 3.0,
                "offset": 0.0,
                "timeline": "default",
                "subject": "default",
            },
            {
                "type": "Word",
                "text": "hello",
                "start": 0.4,
                "duration": 0.3,
                "timeline": "default",
                "subject": "default",
            },
            {
                "type": "Word",
                "text": "world",
                "start": 1.4,
                "duration": 0.2,
                "timeline": "default",
                "subject": "default",
            },
        ]
    )

    shifted = build_offset_events(events, start_offset_s=0.5)

    video_row = shifted.loc[shifted["type"] == "Video"].iloc[0]
    early_word = shifted.loc[shifted["text"] == "hello"].iloc[0]
    later_word = shifted.loc[shifted["text"] == "world"].iloc[0]

    assert float(video_row["start"]) == 0.0
    assert float(video_row["offset"]) == 0.5
    assert float(video_row["duration"]) == 2.5
    assert float(early_word["start"]) == 0.0
    assert float(early_word["duration"]) == 0.2
    assert float(later_word["start"]) == 0.9


def test_build_staggered_prediction_run_interleaves_half_second_segments(tmp_path: Path) -> None:
    base_events = pd.DataFrame([{"type": "Word", "text": "base", "start": 0.0, "duration": 1.0}])
    primary_run = PredictionRun(
        events=base_events,
        preds=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float),
        segments=[
            StoredSegment(0.0, 1.0, "default", [Word(start=0.1, duration=0.2, timeline="default", text="alpha")]),
            StoredSegment(1.0, 1.0, "default", [Word(start=1.1, duration=0.2, timeline="default", text="beta")]),
        ],
        input_kind="video",
        source_path=tmp_path / "clip.mp4",
    )
    staggered_run = PredictionRun(
        events=base_events,
        preds=np.array([[10.0, 0.0], [20.0, 0.0]], dtype=float),
        segments=[
            StoredSegment(0.0, 1.0, "default", [Word(start=0.1, duration=0.2, timeline="default", text="gamma")]),
            StoredSegment(1.0, 1.0, "default", [Word(start=1.1, duration=0.2, timeline="default", text="delta")]),
        ],
        input_kind="video",
        source_path=tmp_path / "clip.mp4",
    )

    merged = build_staggered_prediction_run(primary_run, staggered_run, start_offset_s=0.5)

    np.testing.assert_allclose(
        merged.preds,
        np.array([[1.0, 0.0], [10.0, 0.0], [2.0, 0.0], [20.0, 0.0]], dtype=float),
    )
    assert [segment.start for segment in merged.segments] == [0.0, 0.5, 1.0, 1.5]
    assert merged.run_metadata["sampling_mode"] == "staggered_half_second"
    assert merged.run_metadata["sampling_offset_s"] == 0.5
    assert merged.segments[1].ns_events[0].start == 0.6
