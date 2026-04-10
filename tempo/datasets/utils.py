from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import kagglehub
import soundfile as sf
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}

MANIFEST_COLUMNS = [
    "dataset",
    "subset",
    "sample_id",
    "speaker_id",
    "gender",
    "age",
    "language",
    "transcript",
    "emotion",
    "emotion_original",
    "intensity",
    "split",
    "arousal",
    "valence",
    "dominance",
    "license",
    "audio_path",
    "audio_format",
    "sample_rate_hz",
    "channels",
    "duration_seconds",
    "source_uri",
    "metadata_json",
]

EMOTION_ALIASES = {
    "ang": "anger",
    "anger": "anger",
    "angry": "anger",
    "bored": "boredom",
    "boredom": "boredom",
    "calm": "calm",
    "dis": "disgust",
    "disgust": "disgust",
    "exc": "excitement",
    "excitement": "excitement",
    "excited": "excitement",
    "fear": "fear",
    "fearful": "fear",
    "fea": "fear",
    "fru": "frustration",
    "frustrated": "frustration",
    "frustration": "frustration",
    "hap": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "neu": "neutral",
    "neutral": "neutral",
    "oth": "other",
    "other": "other",
    "pleasant_surprise": "surprise",
    "pleasant_surprised": "surprise",
    "ps": "surprise",
    "sad": "sadness",
    "sadness": "sadness",
    "sur": "surprise",
    "surprised": "surprise",
    "surprise": "surprise",
}

GENDER_ALIASES = {
    "f": "female",
    "female": "female",
    "m": "male",
    "male": "male",
}


def resolve_data_root(data_root: Path | None = None) -> Path:
    return Path(data_root) if data_root else DEFAULT_DATA_ROOT


def raw_dataset_dir(dataset: str, data_root: Path | None = None) -> Path:
    path = resolve_data_root(data_root) / "raw" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dataset_dir(dataset: str, data_root: Path | None = None) -> Path:
    path = resolve_data_root(data_root) / "processed" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_processed_dir(
    dataset: str,
    data_root: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    processed_dir = resolve_data_root(data_root) / "processed" / dataset
    if force and processed_dir.exists():
        shutil.rmtree(processed_dir)
    (processed_dir / "audio").mkdir(parents=True, exist_ok=True)
    return processed_dir


def require_existing_dir(path: Path, message: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(message)
    return path


def list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    audio_files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        audio_files.append(path)
    return sorted(audio_files)


def normalize_emotion(label: str | None) -> str:
    if not label:
        return ""
    cleaned = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return EMOTION_ALIASES.get(cleaned, cleaned)


def normalize_gender(label: str | None) -> str:
    if not label:
        return ""
    return GENDER_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


def mean_or_none(values: list[Any]) -> float | None:
    cleaned: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        cleaned.append(float(value))
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def link_or_copy_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        rel_source = os.path.relpath(source, destination.parent)
        os.symlink(rel_source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def write_audio_bytes(audio_bytes: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(audio_bytes)
    return destination


def project_path_string(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def audio_info(path: Path) -> dict[str, Any]:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return {
            "sample_rate_hz": "",
            "channels": "",
            "duration_seconds": "",
        }

    return {
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "duration_seconds": round(float(info.duration), 6),
    }


def build_record(
    *,
    dataset: str,
    sample_id: str,
    audio_path: Path,
    emotion: str,
    emotion_original: str,
    source_uri: str,
    subset: str = "",
    speaker_id: str = "",
    gender: str = "",
    age: str | int | None = "",
    language: str = "",
    transcript: str = "",
    intensity: str = "",
    split: str = "",
    arousal: float | None = None,
    valence: float | None = None,
    dominance: float | None = None,
    license_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = audio_info(audio_path)
    return {
        "dataset": dataset,
        "subset": subset,
        "sample_id": sample_id,
        "speaker_id": speaker_id,
        "gender": gender,
        "age": "" if age in (None, "") else age,
        "language": language,
        "transcript": transcript,
        "emotion": normalize_emotion(emotion),
        "emotion_original": emotion_original,
        "intensity": intensity,
        "split": split,
        "arousal": "" if arousal is None else round(float(arousal), 6),
        "valence": "" if valence is None else round(float(valence), 6),
        "dominance": "" if dominance is None else round(float(dominance), 6),
        "license": license_name,
        "audio_path": project_path_string(audio_path),
        "audio_format": audio_path.suffix.lower().lstrip("."),
        "sample_rate_hz": info["sample_rate_hz"],
        "channels": info["channels"],
        "duration_seconds": info["duration_seconds"],
        "source_uri": source_uri,
        "metadata_json": json.dumps(metadata or {}, sort_keys=True),
    }


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda row: (row["dataset"], row["subset"], row["sample_id"]))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    return output_path


def combine_manifests(data_root: Path | None = None) -> Path:
    processed_root = resolve_data_root(data_root) / "processed"
    manifests = sorted(
        path
        for path in processed_root.glob("*/manifest.csv")
        if path.is_file()
    )

    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    return write_manifest(rows, processed_root / "manifest.csv")


def download_kaggle_dataset(
    handle: str,
    dataset: str,
    data_root: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    destination = raw_dataset_dir(dataset, data_root)
    if force and destination.exists():
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    kagglehub.dataset_download(
        handle,
        output_dir=str(destination),
        force_download=force,
    )
    return destination


def download_hf_dataset_snapshot(
    repo_id: str,
    dataset: str,
    data_root: Path | None = None,
    *,
    allow_patterns: list[str] | None = None,
    force: bool = False,
) -> Path:
    destination = raw_dataset_dir(dataset, data_root)
    if force and destination.exists():
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )
    return destination


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise ValueError(f"Unsafe tar member detected: {member.name}")
    tar.extractall(destination)


def download_github_repo_tarball(
    repo: str,
    dataset: str,
    data_root: Path | None = None,
    *,
    ref: str = "main",
    force: bool = False,
) -> Path:
    destination = raw_dataset_dir(dataset, data_root)
    if destination.exists() and any(destination.iterdir()) and not force:
        return destination

    if force and destination.exists():
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "repo.tar.gz"
        urllib.request.urlretrieve(url, archive_path)

        extract_root = Path(tmpdir) / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, extract_root)

        extracted_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
        if len(extracted_dirs) != 1:
            raise RuntimeError(f"Expected a single extracted directory for {repo}")

        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(extracted_dirs[0], destination)

    return destination
