from __future__ import annotations

from pathlib import Path

from datasets import Audio, load_dataset

from ..utils import (
    build_record,
    download_hf_dataset_snapshot,
    link_or_copy_audio,
    prepare_processed_dir,
    raw_dataset_dir,
    require_existing_dir,
    write_audio_bytes,
    write_manifest,
)

DATASET = "cameo"
HF_REPO_ID = "amu-cai/CAMEO"
SOURCE_URI = "https://huggingface.co/datasets/amu-cai/CAMEO"


def download_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    return download_hf_dataset_snapshot(
        HF_REPO_ID,
        DATASET,
        data_root,
        allow_patterns=["README.md", "data/*.parquet"],
        force=force,
    )


def preprocess_dataset(data_root: Path | None = None, *, force: bool = False) -> Path:
    raw_dir = require_existing_dir(
        raw_dataset_dir(DATASET, data_root),
        "CAMEO has not been downloaded yet. Run download_dataset() first.",
    )
    processed_dir = prepare_processed_dir(DATASET, data_root, force=force)
    parquet_dir = require_existing_dir(
        raw_dir / "data",
        "CAMEO parquet files are missing. Run download_dataset() first.",
    )

    records = []
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        dataset_rows = load_dataset("parquet", data_files=str(parquet_path), split="train")
        dataset_rows = dataset_rows.cast_column("audio", Audio(decode=False))
        subset_name = parquet_path.stem.split("-", 1)[0]

        for row in dataset_rows:
            audio_payload = row.get("audio") or {}
            file_id = str(row.get("file_id", "")).strip()
            if not file_id:
                continue

            output_name = Path(audio_payload.get("path") or file_id).name
            destination = processed_dir / "audio" / subset_name / output_name

            audio_bytes = audio_payload.get("bytes")
            if audio_bytes:
                write_audio_bytes(bytes(audio_bytes), destination)
            else:
                source_audio_path = audio_payload.get("path")
                if not source_audio_path:
                    continue
                link_or_copy_audio(Path(source_audio_path), destination)

            records.append(
                build_record(
                    dataset=DATASET,
                    subset=str(row.get("dataset", subset_name)).strip().lower(),
                    sample_id=f"{subset_name}__{Path(file_id).stem}",
                    audio_path=destination,
                    emotion=str(row.get("emotion", "")).strip(),
                    emotion_original=str(row.get("emotion", "")).strip(),
                    source_uri=SOURCE_URI,
                    speaker_id=str(row.get("speaker_id", "")).strip(),
                    gender=str(row.get("gender", "")).strip().lower(),
                    age=str(row.get("age", "")).strip(),
                    language=str(row.get("language", "")).strip().lower(),
                    transcript=str(row.get("transcription", "")).strip(),
                    license_name=str(row.get("license", "")).strip(),
                    metadata={
                        "file_id": file_id,
                    },
                )
            )

    return write_manifest(records, processed_dir / "manifest.csv")
