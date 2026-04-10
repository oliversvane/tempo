from __future__ import annotations

from pathlib import Path

from ..utils import (
    build_record,
    download_kaggle_dataset,
    link_or_copy_audio,
    list_audio_files,
    normalize_emotion,
    prepare_processed_dir,
    raw_dataset_dir,
    require_existing_dir,
    write_manifest,
)

DATASET = "tess"
KAGGLE_HANDLE = "ejlok1/toronto-emotional-speech-set-tess"
SOURCE_URI = "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "TESS has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        sample_id = source_path.stem
        if sample_id in seen_sample_ids:
            continue

        speaker_group = ""
        emotion_original = ""
        if "_" in source_path.parent.name:
            speaker_group, emotion_original = source_path.parent.name.split("_", 1)
        elif "_" in sample_id:
            speaker_group = sample_id.split("_", 1)[0]
            emotion_original = sample_id.rsplit("_", 1)[-1]

        emotion = normalize_emotion(emotion_original)
        if not emotion:
            continue

        seen_sample_ids.add(sample_id)
        destination = link_or_copy_audio(source_path, processed_dir / "audio" / source_path.name)
        age = ""
        if speaker_group.startswith("O"):
            age = "older_adult"
        elif speaker_group.startswith("Y"):
            age = "young_adult"

        records.append(
            build_record(
                dataset=DATASET,
                sample_id=sample_id,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_original,
                source_uri=SOURCE_URI,
                speaker_id=speaker_group,
                gender="female",
                age=age,
                language="english",
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

