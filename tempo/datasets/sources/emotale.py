from __future__ import annotations

import csv
from pathlib import Path

from ..utils import (
    build_record,
    download_github_repo_tarball,
    link_or_copy_audio,
    list_audio_files,
    mean_or_none,
    normalize_gender,
    prepare_processed_dir,
    raw_dataset_dir,
    require_existing_dir,
    write_manifest,
)

DATASET = "emotale"
GITHUB_REPO = "snehadas/EmoTale"
SOURCE_URI = "https://github.com/snehadas/EmoTale/tree/main/"

EMOTION_MAP = {
    "A": "anger",
    "B": "boredom",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
}

LANGUAGE_MAP = {
    "DK": "danish",
    "EN": "english",
}


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_github_repo_tarball(GITHUB_REPO, DATASET, data_root, force=force)


def _load_speaker_info(raw_dir: Path) -> dict[str, dict[str, str]]:
    speaker_info_path = raw_dir / "speaker_info.csv"
    if not speaker_info_path.exists():
        return {}

    speakers: dict[str, dict[str, str]] = {}
    with speaker_info_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            speaker_id = row.get("speaker", "").strip()
            if speaker_id:
                speakers[speaker_id] = row
    return speakers


def _load_annotations(raw_dir: Path) -> dict[str, dict[str, str]]:
    annotations_path = raw_dir / "annotations.csv"
    if not annotations_path.exists():
        return {}

    annotations: dict[str, dict[str, str]] = {}
    with annotations_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            file_name = row.get("file", "").strip()
            if file_name:
                annotations[file_name] = row
    return annotations


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "EmoTale has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)
    speaker_info = _load_speaker_info(raw_dir)
    annotations = _load_annotations(raw_dir)

    records = []
    for source_path in list_audio_files(raw_dir):
        parts = source_path.stem.split("_")
        if len(parts) != 4:
            continue

        language_code, speaker_id, emotion_code, sentence_id = parts
        emotion_original = emotion_code
        annotation = annotations.get(source_path.name)
        if annotation and annotation.get("gt_emotion"):
            emotion_original = annotation["gt_emotion"].strip().upper()

        emotion = EMOTION_MAP.get(emotion_original)
        if not emotion:
            continue

        speaker = speaker_info.get(speaker_id, {})
        destination = link_or_copy_audio(source_path, processed_dir / "audio" / source_path.name)
        records.append(
            build_record(
                dataset=DATASET,
                sample_id=source_path.stem,
                audio_path=destination,
                emotion=emotion,
                emotion_original=emotion_original,
                source_uri=SOURCE_URI,
                speaker_id=speaker_id,
                gender=normalize_gender(speaker.get("gender")),
                age=speaker.get("age", ""),
                language=LANGUAGE_MAP.get(language_code.upper(), language_code.lower()),
                arousal=mean_or_none(
                    [
                        annotation.get("a1_A") if annotation else None,
                        annotation.get("a2_A") if annotation else None,
                        annotation.get("a3_A") if annotation else None,
                    ]
                ),
                valence=mean_or_none(
                    [
                        annotation.get("a1_V") if annotation else None,
                        annotation.get("a2_V") if annotation else None,
                        annotation.get("a3_V") if annotation else None,
                    ]
                ),
                dominance=mean_or_none(
                    [
                        annotation.get("a1_D") if annotation else None,
                        annotation.get("a2_D") if annotation else None,
                        annotation.get("a3_D") if annotation else None,
                    ]
                ),
                metadata={
                    "sentence_id": sentence_id,
                    "language_code": language_code.upper(),
                    "annotator_categories": [
                        annotation.get("a1_cat") if annotation else "",
                        annotation.get("a2_cat") if annotation else "",
                        annotation.get("a3_cat") if annotation else "",
                    ],
                },
            )
        )

    return write_manifest(records, processed_dir / "manifest.csv")

