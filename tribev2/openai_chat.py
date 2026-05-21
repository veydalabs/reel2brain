from __future__ import annotations

import json
import logging
import typing as tp

import pandas as pd

from tribev2.easy import (
    build_comparison_display_reference,
    ImageComparisonRun,
    MultiModalRun,
    PredictionRun,
    build_emotion_hypothesis_frame,
    build_run_roi_frame,
    build_run_zone_frame,
    build_selected_timestep_roi_frame,
    build_timestep_zone_frame,
    collect_timestep_metadata,
    collect_run_text,
    render_run_panel_bytes,
    summarize_predictions,
)

render_brain_panel_bytes = render_run_panel_bytes

DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.4"
LOGGER = logging.getLogger("tribev2.openai_chat")

COMMON_UNCERTAINTIES = [
    "the exact image ordering if the display does not strictly follow t0->tN or if only key frames were attached",
    "strict comparability of visual intensities without a shared colorbar or confirmation of identical normalization",
    "fine-grained anatomical attribution to one precise region rather than a larger cortical system",
    "any strong neuroscientific interpretation, because these are TRIBE v2 predictions rather than real fMRI measurements",
]


def _build_pipeline_summary() -> list[str]:
    return [
        "The Reel2Brain app converts the raw source into events or temporal segments.",
        "Foundation encoders extract video, audio, and text representations depending on the available modality.",
        "TRIBE v2 projects those representations into predicted cortical activity on the brain surface.",
        "Each timestep corresponds to a predicted segment of the stimulus, not a recorded brain measurement.",
    ]


def _build_modality_notes(run: PredictionRun | ImageComparisonRun) -> list[str]:
    if isinstance(run, ImageComparisonRun):
        if run.compare_kind == "image":
            return [
                "This run compares two images processed independently and displayed side by side.",
                "Each still image is converted into a short silent static video clip to pass through TRIBE v2.",
                "Timesteps for a single image do not represent real temporal evolution; they repeat the same visual content.",
                "The comparison should focus mainly on spatial distribution, laterality, salience, and the plausible affective tone of the visual content.",
            ]
        if run.compare_kind == "video":
            return [
                "This run compares two videos processed independently and displayed side by side.",
                "Each column corresponds to a distinct video run with its own timeline.",
                "The comparison should connect cortical pattern differences to the visual, acoustic, and verbal content of each video.",
            ]
        if run.compare_kind == "audio":
            return [
                "This run compares two audio tracks processed independently and displayed side by side.",
                "Each column corresponds to a distinct audio run with its own timeline.",
                "The comparison should focus mainly on prosody, tension, rhythm, timbre, and any aligned text.",
            ]
        return [
            "This run compares two texts processed independently and displayed side by side.",
            "Each column corresponds to a distinct text run with its own aligned or synthetic timeline.",
            "The comparison should connect cortical pattern differences to lexical content, tone, and structure across the two texts.",
        ]
    if isinstance(run, MultiModalRun) or run.input_kind == "multimodal":
        return [
            "The source combines several modalities in a single run.",
            "The app computes one fused run for the main prediction, then separate modality runs to color the brain by contribution.",
            "The overlay color code is: red for visual, green for audio, blue for text.",
            "The colors represent relative contribution by modality family at the same timestep, not different anatomical regions.",
        ]
    if run.input_kind == "video":
        return [
            "The source is a video. Each timestep corresponds to a temporal segment of the clip.",
            "Interpretation can link maps to visual content, rhythm, speech, and sound when those are present in the segment.",
            "Emotions or feelings should be framed as plausible hypotheses about the stimulus, not as direct readings of a mental state.",
        ]
    if run.input_kind == "audio":
        return [
            "The source is audio. Each timestep corresponds to a sound segment of the file.",
            "If aligned text exists, it acts as a supporting cue, but the main source remains the acoustic signal.",
            "Interpretation can comment on plausible prosody, tension, calm, threat, joy, or sadness by tying them back to the sound stimulus.",
        ]
    if run.input_kind == "text":
        return [
            "The source is text. In this fork, direct text mode builds synthetic timings word by word or segment by segment.",
            "Timesteps therefore reflect an artificial segmentation of the text, not a real audio or video recording.",
            "Emotions or feelings can be inferred from lexical field and tone, then compared cautiously against the predicted map.",
        ]
    if run.input_kind == "image":
        return [
            "The source is a still image.",
            "In this fork, the image is converted into a short silent static video clip to pass through TRIBE v2.",
            "Timesteps represent repeated passes over the same visual content, not real temporal narration.",
            "Interpretation should focus mainly on visual structure, spatial salience, and the plausible affective tone of the image content.",
        ]
    return [
        "The modality should be read from the attached context.",
        "Timesteps correspond to segments predicted by TRIBE v2.",
    ]


def _build_interpretation_contract(run: PredictionRun | ImageComparisonRun) -> dict[str, tp.Any]:
    comparison_hint = (
        f"If two {run.compare_kind}s or two runs are present, compare {run.compare_kind} 1 vs {run.compare_kind} 2 explicitly, or run 1 vs run 2."
        if isinstance(run, ImageComparisonRun)
        else "If the user compares several moments, clearly call out pattern differences from one timestep to another."
    )
    return {
        "mission": [
            "Explain what the predicted cortical activity map is showing.",
            "State what the pattern is typically associated with: vision, audition, language, attention, salience, frontal control, social cognition, or plausible affective load.",
            "Prioritize the HCP zone tables and the affective-hypothesis radar when they are provided.",
            "If the user asks, offer a cautious reading of valence or plausible emotions: positive, negative, mixed, fear, desire, joy, sadness, anger, calm, tension, threat, surprise.",
            "Always distinguish observation, hypothesis, and uncertainty.",
        ],
        "format_attendu": [
            "1. What we see",
            "2. What it is typically associated with",
            "3. Plausible emotion or feeling",
            "4. Cortical zones and stimulus cues that support that reading",
            "5. What remains uncertain",
        ],
        "regles": [
            "Start from the modality and remind the reader what a timestep represents in this run.",
            "Never talk as if TRIBE v2 directly measures brain activity: these are model predictions.",
            "Do not turn a plausible emotion into a certainty.",
            "The visible output here is an fsaverage5 cortical surface; do not claim to directly observe the amygdala, deep insula, or other subcortical structures unless they are explicitly provided.",
            "The emotional radar is a synthesis heuristic, not a clinical emotion decoder.",
            comparison_hint,
            "Respond in English unless the user explicitly requests another language.",
        ],
        "incertitudes_a_mentionner": COMMON_UNCERTAINTIES,
    }


def build_chat_system_prompt(run: PredictionRun | ImageComparisonRun) -> str:
    sections = [
        "You are the analysis assistant for the Reel2Brain app.",
        "Your task is to explain the model outputs from the numeric data, zone tables, affective-hypothesis radar, and timestep images that are provided to you.",
        "",
        "How the TRIBE v2 workflow operates in this app:",
        *[f"- {item}" for item in _build_pipeline_summary()],
        "- The zone tables aggregate the fsaverage5 cortical surface by HCP-MMP ROIs. They cover cortex only.",
        "- The emotional radar combines stimulus cues, dominant cortical zones, and modality. It is not a direct readout of a mental state.",
        "- If the JSON context indicates `display_normalization = shared_percentile_99_reference` or a shared variant, visual intensities are comparable between runs on the same display scale.",
        "",
        "How to read the current modality:",
        *[f"- {item}" for item in _build_modality_notes(run)],
        "",
        "Interpretation frame:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["mission"]],
        "",
        "Expected response format:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["format_attendu"]],
        "",
        "Rules:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["regles"]],
        "",
        "What remains uncertain:",
        *[f"- {item}" for item in COMMON_UNCERTAINTIES],
        "",
        "Do not:",
        "- give medical diagnoses",
        "- claim direct access to intentions or mental state",
        "- claim subcortical structures that are not visible in the cortical data",
        "- make overly fine anatomical claims without evidence",
        "- present strong neuroscientific interpretations as certain",
    ]
    return "\n".join(sections).strip()


def build_raw_timestep_frame(run: PredictionRun) -> pd.DataFrame:
    """Build a raw per-timestep table without hardcoded interpretive labels."""
    summary = summarize_predictions(run.preds)
    metadata = collect_timestep_metadata(run)
    rows: list[dict[str, tp.Any]] = []
    for idx, row in summary.iterrows():
        meta = metadata[idx] if idx < len(metadata) else {}
        rows.append(
            {
                "timestep": int(row["timestep"]),
                "start_s": round(float(meta.get("start", idx)), 3),
                "duration_s": round(float(meta.get("duration", 1.0)), 3),
                "text": str(meta.get("text", "") or ""),
                "mean": round(float(row["mean"]), 6),
                "std": round(float(row["std"]), 6),
                "mean_abs": round(float(row["mean_abs"]), 6),
                "max_abs": round(float(row["max_abs"]), 6),
            }
        )
    return pd.DataFrame(rows)


def _truncate_text(value: str | None, limit: int = 1800) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _select_key_timestep_indices(frame: pd.DataFrame, max_images: int = 4) -> list[int]:
    if frame.empty:
        return []
    candidates = [
        int(frame.iloc[0]["timestep"]),
        int(frame.iloc[len(frame) // 2]["timestep"]),
        int(frame.iloc[-1]["timestep"]),
        int(frame.sort_values("mean_abs", ascending=False).iloc[0]["timestep"]),
    ]
    top_dynamic = frame.sort_values("mean_abs", ascending=False)["timestep"].astype(int).tolist()
    candidates.extend(top_dynamic[: max(0, max_images * 2)])
    selected: list[int] = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_images:
            break
    return selected


def _to_base64_data_url(raw_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    import base64

    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('ascii')}"


def _prediction_run_context_images(
    run: PredictionRun,
    *,
    run_label: str,
    image_detail: str,
    max_images: int,
    display_reference: object | None = None,
) -> tuple[dict[str, tp.Any], list[dict[str, str]], list[str]]:
    frame = build_raw_timestep_frame(run)
    selected = _select_key_timestep_indices(frame, max_images=max_images)
    rows_for_prompt = frame.sort_values("mean_abs", ascending=False).head(8)
    modality_notes = _build_modality_notes(run)
    zone_payload = {
        "run_zone_summary": build_run_zone_frame(run).to_dict(orient="records"),
        "run_top_rois": build_run_roi_frame(run, top_k=24).to_dict(orient="records"),
        "zone_timeseries": build_timestep_zone_frame(run).to_dict(orient="records"),
        "selected_timestep_rois": build_selected_timestep_roi_frame(
            run,
            indices=selected,
            top_k=10,
        ).to_dict(orient="records"),
        "emotion_hypotheses": build_emotion_hypothesis_frame(run).to_dict(orient="records"),
    }
    payload = {
        "run_label": run_label,
        "input_kind": run.input_kind,
        "n_timesteps": int(len(run.preds)),
        "n_vertices": int(run.preds.shape[1]),
        "source_path": str(run.source_path) if run.source_path is not None else None,
        "raw_text_excerpt": _truncate_text(run.raw_text),
        "full_text_excerpt": _truncate_text(collect_run_text(run), limit=3200),
        "tribev2_pipeline_summary": _build_pipeline_summary(),
        "modality_notes": modality_notes,
        "cortical_zone_notes": [
            "The HCP zone and ROI tables come from the fsaverage5 cortex and do not cover subcortical structures.",
            "Zone values use aggregated mean absolute amplitude, which is useful for comparing relative patterns within a run.",
            "Emotional hypotheses remain cautious and should be read as stimulus hypotheses, not as certainty about a subject's internal state.",
        ],
        "interpretation_contract": _build_interpretation_contract(run),
        "selected_timestep_image_policy": (
            "The attached images are key timesteps selected automatically from the start, middle, end, and mean_abs peaks."
        ),
        "display_normalization": (
            "shared_percentile_99_reference"
            if display_reference is not None
            else "per_run_percentile_99_reference"
        ),
        "timestep_rows": rows_for_prompt.to_dict(orient="records"),
        "zone_payload": zone_payload,
    }
    image_parts: list[dict[str, str]] = []
    labels: list[str] = []
    for idx in selected:
        row = frame.loc[frame["timestep"] == idx].iloc[0]
        labels.append(
            f"{run_label}: timestep {idx} ({float(row['start_s']):.2f}s, mean_abs={float(row['mean_abs']):.4f})"
        )
        image_parts.append(
            {
                "type": "input_image",
                "detail": image_detail,
                "image_url": _to_base64_data_url(
                    render_run_panel_bytes(
                        run,
                        timestep=idx,
                        image_format="JPEG",
                        quality=84,
                        display_reference=tp.cast(tp.Any, display_reference),
                    )
                ),
            }
        )
    return payload, image_parts, labels


def build_openai_context_bundle(
    run: PredictionRun | ImageComparisonRun,
    *,
    image_detail: str = "low",
    max_images: int = 4,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Prepare the multimodal context sent to OpenAI."""
    if isinstance(run, ImageComparisonRun):
        per_run_limit = max(1, max_images // max(len(run.runs), 1))
        display_reference = build_comparison_display_reference(run)
        payload_runs = []
        image_parts: list[dict[str, str]] = []
        labels: list[str] = []
        for idx, item in enumerate(run.runs, start=1):
            payload, parts, run_labels = _prediction_run_context_images(
                item,
                run_label=f"{run.compare_kind}_{idx}",
                image_detail=image_detail,
                max_images=per_run_limit,
                display_reference=display_reference,
            )
            payload_runs.append(payload)
            image_parts.extend(parts)
            labels.extend(run_labels)
        context = {
            "kind": f"tribev2_{run.compare_kind}_comparison",
            "compare_kind": run.compare_kind,
            "n_runs": len(run.runs),
            "display_normalization": "shared_percentile_99_reference_across_all_compared_runs",
            "tribev2_pipeline_summary": _build_pipeline_summary(),
            "comparison_notes": _build_modality_notes(run),
            "interpretation_contract": _build_interpretation_contract(run),
            "runs": payload_runs,
        }
        return json.dumps(context, ensure_ascii=False, indent=2), image_parts[:max_images], labels[:max_images]

    payload, image_parts, labels = _prediction_run_context_images(
        run,
        run_label="run",
        image_detail=image_detail,
        max_images=max_images,
    )
    context = {
        "kind": "tribev2_prediction_run",
        "run": payload,
    }
    return json.dumps(context, ensure_ascii=False, indent=2), image_parts, labels


def extract_response_text(response: tp.Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            candidate = getattr(content, "text", None)
            if candidate:
                chunks.append(str(candidate))
    joined = "\n\n".join(part.strip() for part in chunks if str(part).strip())
    return joined or "Aucune reponse textuelle renvoyee par l'API."


def request_openai_run_explanation(
    *,
    api_key: str,
    model: str,
    reasoning_effort: str,
    user_prompt: str,
    run: PredictionRun | ImageComparisonRun,
    previous_response_id: str | None = None,
    include_context: bool = True,
    image_detail: str = "low",
    max_images: int = 4,
    context_bundle: tuple[str, list[dict[str, str]], list[str]] | None = None,
) -> tuple[str, str | None, list[str]]:
    """Send the current run plus the user prompt to OpenAI Responses API."""
    from openai import OpenAI

    LOGGER.info(
        "OpenAI request start | model=%s | include_context=%s | max_images=%s | comparison=%s",
        model,
        include_context,
        max_images,
        isinstance(run, ImageComparisonRun),
    )
    client = OpenAI(api_key=api_key)
    labels: list[str] = []
    content: list[dict[str, str]] = []
    input_items: list[dict[str, tp.Any]] = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": build_chat_system_prompt(run)}],
        }
    ]
    if include_context:
        if context_bundle is None:
            context_bundle = build_openai_context_bundle(
                run,
                image_detail=image_detail,
                max_images=max_images,
            )
        context_text, image_parts, labels = context_bundle
        content.append(
            {
                "type": "input_text",
                "text": (
                    "Contexte TRIBE v2 du run courant:\n"
                    + context_text
                    + "\n\nUtilise ce contexte et les images jointes pour repondre a la question suivante."
                ),
            }
        )
        content.extend(image_parts)
    content.append({"type": "input_text", "text": user_prompt})
    input_items.append({"role": "user", "content": content})
    response = client.responses.create(
        model=model,
        previous_response_id=previous_response_id,
        reasoning={"effort": reasoning_effort},
        input=input_items,
    )
    LOGGER.info(
        "OpenAI request complete | model=%s | response_id=%s",
        model,
        getattr(response, "id", None),
    )
    return extract_response_text(response), getattr(response, "id", None), labels
