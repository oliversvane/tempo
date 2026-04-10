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

DATASET = "savee"
KAGGLE_HANDLE = "ejlok1/surrey-audiovisual-expressed-emotion-savee"
SOURCE_URI = "https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee"

EMOTION_MAP = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}

STEM_PATTERN = re.compile(r"^(?P<speaker>[A-Za-z]+)_(?P<emotion>[a-z]+)(?P<index>\d+)$")


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_kaggle_dataset(KAGGLE_HANDLE, DATASET, data_root, force=force)


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "SAVEE has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)

    records = []
    seen_sample_ids: set[str] = set()
    for source_path in list_audio_files(raw_dir):
        match = STEM_PATTERN.match(source_path.stem)
        if not match or source_path.stem in seen_sample_ids:
            continue

        emotion_code = match.group("emotion").lower()
        emotion = EMOTION_MAP.get(emotion_code)
        if not emotion:
            continue

        seen_sample_ids.add(source_path.stem)
        destination = link_or_copy_audio(source_path, processed_dir / "audio" / source_path.name)
        records.append(
            build_record(
                dataset=DATASET,
                sample_id=source_path.stem,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_code,
                source_uri=SOURCE_URI,
                speaker_id=match.group("speaker").upper(),
                gender="male",
                language="english",
                metadata={
                    "clip_index": int(match.group("index")),
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

