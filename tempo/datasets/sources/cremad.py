from __future__ import annotations

import csv
from pathlib import Path

from ..utils import (
    build_record,
    download_kaggle_dataset,
    link_or_copy_audio,
    list_audio_files,
    normalize_gender,
    prepare_processed_dir,
    raw_dataset_dir,
    require_existing_dir,
    write_manifest,
)

DATASET = "cremad"
KAGGLE_HANDLE = "ejlok1/cremad"
SOURCE_URI = "https://www.kaggle.com/datasets/ejlok1/cremad"

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness",
}

INTENSITY_MAP = {
    "HI": "high",
    "LO": "low",
    "MD": "medium",
    "XX": "unspecified",
}


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def _load_demographics(raw_dir: Path) -> dict[str, dict[str, str]]:
    csv_candidates = sorted(
        path for path in raw_dir.rglob("*.csv") if path.name.lower() == "videodemographics.csv"
    )
    if not csv_candidates:
        return {}

    rows: dict[str, dict[str, str]] = {}
    with csv_candidates[0].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            actor_id = str(row.get("ActorID", "")).strip()
            if actor_id:
                rows[actor_id] = row
    return rows


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "CREMA-D has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)
    demographics = _load_demographics(raw_dir)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        parts = source_path.stem.split("_")
        if len(parts) != 4:
            continue

        speaker_id, sentence_code, emotion_code, intensity_code = parts
        if source_path.stem in seen_sample_ids:
            continue
        emotion = EMOTION_MAP.get(emotion_code.upper())
        if not emotion:
            continue

        seen_sample_ids.add(source_path.stem)
        speaker_meta = demographics.get(speaker_id, {})
        destination = link_or_copy_audio(source_path, processed_dir / "audio" / source_path.name)
        records.append(
            build_record(
                dataset=DATASET,
                sample_id=source_path.stem,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_code.upper(),
                source_uri=SOURCE_URI,
                speaker_id=speaker_id,
                gender=normalize_gender(speaker_meta.get("Sex")),
                age=speaker_meta.get("Age", ""),
                language="english",
                intensity=INTENSITY_MAP.get(intensity_code.upper(), intensity_code.lower()),
                metadata={
                    "sentence_code": sentence_code,
                    "race": speaker_meta.get("Race", ""),
                    "ethnicity": speaker_meta.get("Ethnicity", ""),
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

