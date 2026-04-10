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

DATASET = "emodb"
KAGGLE_HANDLE = "piyushagni5/berlin-database-of-emotional-speech-emodb"
SOURCE_URI = "https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb"

EMOTION_MAP = {
    "A": "fear",
    "E": "disgust",
    "F": "happiness",
    "L": "boredom",
    "N": "neutral",
    "T": "sadness",
    "W": "anger",
}


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "EMO-DB has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        sample_id = source_path.stem
        if len(sample_id) < 2 or sample_id in seen_sample_ids:
            continue

        emotion_code = sample_id[-2].upper()
        emotion = EMOTION_MAP.get(emotion_code)
        if not emotion:
            continue

        seen_sample_ids.add(sample_id)
        destination = link_or_copy_audio(source_path, processed_dir / "audio" / source_path.name)
        records.append(
            build_record(
                dataset=DATASET,
                sample_id=sample_id,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_code,
                source_uri=SOURCE_URI,
                speaker_id=sample_id[:2],
                language="german",
                metadata={
                    "utterance_code": sample_id[2:5],
                    "version_code": sample_id[-1],
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

