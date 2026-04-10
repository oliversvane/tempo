from __future__ import annotations

import re
from pathlib import Path

from ..utils import (
    build_record,
    download_kaggle_dataset,
    link_or_copy_audio,
    list_audio_files,
    prepare_processed_dir,
    raw_dataset_dir,
    require_existing_dir,
    write_manifest,
)

DATASET = "iemocap"
KAGGLE_HANDLE = "jamaliasultanajisha/iemocap-full"
SOURCE_URI = "https://www.kaggle.com/datasets/jamaliasultanajisha/iemocap-full"

EMOTION_MAP = {
    "ang": "anger",
    "dis": "disgust",
    "exc": "excitement",
    "fea": "fear",
    "fru": "frustration",
    "hap": "happiness",
    "neu": "neutral",
    "oth": "other",
    "sad": "sadness",
    "sur": "surprise",
}

EVAL_LINE = re.compile(
    r"^\[(?P<start>[\d.]+) - (?P<end>[\d.]+)\]\t(?P<utterance>[^\t]+)\t(?P<label>[^\t]+)\t\[(?P<vad>[^\]]+)\]"
)
TRANSCRIPT_LINE = re.compile(r"^(?P<utterance>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def _load_labels(raw_dir: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for path in sorted(raw_dir.rglob("*.txt")):
        if "EmoEvaluation" not in path.parts:
            continue
        session = next((part for part in path.parts if part.startswith("Session")), "")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = EVAL_LINE.match(line.strip())
            if not match:
                continue
            utterance = match.group("utterance")
            label = match.group("label").strip().lower()
            if label == "xxx":
                continue
            labels[utterance] = {
                "emotion_original": label,
                "session": session,
                "start": match.group("start"),
                "end": match.group("end"),
                "vad": match.group("vad"),
            }
    return labels


def _load_transcripts(raw_dir: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in sorted(raw_dir.rglob("*.txt")):
        if "transcriptions" not in path.parts:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = TRANSCRIPT_LINE.match(line.strip())
            if not match:
                continue
            transcripts[match.group("utterance")] = match.group("text").strip()
    return transcripts


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "IEMOCAP has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)
    labels = _load_labels(raw_dir)
    transcripts = _load_transcripts(raw_dir)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        sample_id = source_path.stem
        label_info = labels.get(sample_id)
        if not label_info or sample_id in seen_sample_ids:
            continue

        emotion = EMOTION_MAP.get(label_info["emotion_original"])
        if not emotion:
            continue

        seen_sample_ids.add(sample_id)
        utterance_suffix = sample_id.rsplit("_", 1)[-1]
        gender = "female" if utterance_suffix.startswith("F") else "male" if utterance_suffix.startswith("M") else ""
        dialogue_id = sample_id.rsplit("_", 1)[0]
        session = label_info["session"]

        destination = link_or_copy_audio(
            source_path,
            processed_dir / "audio" / session / source_path.name,
        )
        records.append(
            build_record(
                dataset=DATASET,
                subset=session.lower(),
                sample_id=sample_id,
                audio_path=destination,
                emotion=emotion,
                emotion_original=label_info["emotion_original"],
                source_uri=SOURCE_URI,
                speaker_id=f"{dialogue_id.split('_')[0]}_{utterance_suffix[0]}",
                gender=gender,
                language="english",
                transcript=transcripts.get(sample_id, ""),
                metadata={
                    "dialogue_id": dialogue_id,
                    "annotation_start_seconds": float(label_info["start"]),
                    "annotation_end_seconds": float(label_info["end"]),
                    "vad": label_info["vad"],
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

