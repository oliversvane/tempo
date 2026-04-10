from __future__ import annotations

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

DATASET = "ravdess"
KAGGLE_HANDLE = "uwrfkaggler/ravdess-emotional-speech-audio"
SOURCE_URI = "https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "RAVDESS has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        parts = source_path.stem.split("-")
        if len(parts) != 7 or source_path.stem in seen_sample_ids:
            continue

        modality, vocal_channel, emotion_code, intensity_code, statement_code, repetition_code, actor_code = parts
        if modality != "03" or vocal_channel != "01":
            continue

        emotion = EMOTION_MAP.get(emotion_code)
        if not emotion:
            continue

        seen_sample_ids.add(source_path.stem)
        gender = "male" if int(actor_code) % 2 == 1 else "female"
        destination = link_or_copy_audio(
            source_path,
            processed_dir / "audio" / f"Actor_{actor_code}" / source_path.name,
        )
        records.append(
            build_record(
                dataset=DATASET,
                sample_id=source_path.stem,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_code,
                source_uri=SOURCE_URI,
                speaker_id=actor_code,
                gender=gender,
                language="english",
                intensity=INTENSITY_MAP.get(intensity_code, intensity_code),
                metadata={
                    "statement_code": statement_code,
                    "repetition_code": repetition_code,
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

