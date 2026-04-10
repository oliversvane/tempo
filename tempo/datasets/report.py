from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .utils import resolve_data_root

UNKNOWN_VALUES = {"", "unknown", "<unk>", "none", "null", "nan"}
BASIC_EMOTIONS = {
    "anger",
    "boredom",
    "calm",
    "disgust",
    "excitement",
    "fear",
    "frustration",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
    "other",
}

PLOT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def write_report(source: str = "all", data_root: Path | None = None) -> tuple[Path, Path, list[Path]]:
    data_root = resolve_data_root(data_root)
    manifest_path = _manifest_path(source, data_root)
    rows = _read_manifest(manifest_path)
    summary = _build_summary(rows, source=source, manifest_path=manifest_path)

    markdown_path, json_path = _report_paths(source, data_root)
    plot_metadata = _write_plots(summary, rows, source, data_root)
    summary["artifacts"] = {
        "plots": plot_metadata,
    }
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return markdown_path, json_path, [Path(item["absolute_path"]) for item in plot_metadata]


def _manifest_path(source: str, data_root: Path) -> Path:
    if source == "all":
        manifest_path = data_root / "processed" / "manifest.csv"
    else:
        manifest_path = data_root / "processed" / source / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run preprocessing before generating a report."
        )
    return manifest_path


def _report_paths(source: str, data_root: Path) -> tuple[Path, Path]:
    base_dir = _report_base_dir(source, data_root)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "report.md", base_dir / "report.json"


def _report_base_dir(source: str, data_root: Path) -> Path:
    if source == "all":
        return data_root / "processed"
    return data_root / "processed" / source


def _plot_dir(source: str, data_root: Path) -> Path:
    plot_dir = _report_base_dir(source, data_root) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bucket(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    return "unknown" if cleaned in UNKNOWN_VALUES else cleaned


def _gender_bucket(value: str | None) -> str:
    cleaned = _bucket(value)
    if cleaned in {"female", "male", "child", "other"}:
        return cleaned
    return "unknown"


def _duration_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "total_seconds": 0.0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }

    return {
        "total_seconds": round(sum(values), 6),
        "mean_seconds": round(statistics.mean(values), 6),
        "median_seconds": round(statistics.median(values), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def _counter_rows(
    counter: Counter[str],
    total_rows: int,
    duration_by_key: dict[str, float],
    dataset_counter_by_key: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common():
        item: dict[str, Any] = {
            "label": key,
            "samples": count,
            "share_percent": round((count / total_rows) * 100, 2) if total_rows else 0.0,
            "hours": round(duration_by_key.get(key, 0.0) / 3600, 3),
        }
        if dataset_counter_by_key is not None:
            item["datasets"] = sorted(dataset_counter_by_key.get(key, set()))
        rows.append(item)
    return rows


def _top_labels(counter: Counter[str], limit: int = 5) -> str:
    items = counter.most_common(limit)
    if not items:
        return "-"
    return ", ".join(f"{label} ({count})" for label, count in items)


def _build_summary(
    rows: list[dict[str, str]],
    *,
    source: str,
    manifest_path: Path,
) -> dict[str, Any]:
    total_rows = len(rows)
    durations = [_safe_float(row.get("duration_seconds")) for row in rows]
    duration_values = [value for value in durations if value is not None]

    language_counter: Counter[str] = Counter()
    gender_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    sample_rate_counter: Counter[str] = Counter()
    channels_counter: Counter[str] = Counter()
    audio_format_counter: Counter[str] = Counter()

    duration_by_emotion: defaultdict[str, float] = defaultdict(float)
    duration_by_language: defaultdict[str, float] = defaultdict(float)
    dataset_counter_by_emotion: defaultdict[str, set[str]] = defaultdict(set)
    dataset_counter_by_language: defaultdict[str, set[str]] = defaultdict(set)

    transcript_count = 0
    age_count = 0
    gender_known_count = 0
    arousal_count = 0
    valence_count = 0
    dominance_count = 0
    dataset_scoped_speakers: set[tuple[str, str]] = set()

    per_dataset_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        dataset = row["dataset"]
        per_dataset_rows[dataset].append(row)

        duration_seconds = _safe_float(row.get("duration_seconds")) or 0.0
        language = _bucket(row.get("language"))
        gender = _gender_bucket(row.get("gender"))
        emotion = _bucket(row.get("emotion"))
        sample_rate = row.get("sample_rate_hz") or "unknown"
        channels = row.get("channels") or "unknown"
        audio_format = _bucket(row.get("audio_format"))

        language_counter[language] += 1
        gender_counter[gender] += 1
        emotion_counter[emotion] += 1
        sample_rate_counter[sample_rate] += 1
        channels_counter[channels] += 1
        audio_format_counter[audio_format] += 1

        duration_by_language[language] += duration_seconds
        duration_by_emotion[emotion] += duration_seconds
        dataset_counter_by_language[language].add(dataset)
        dataset_counter_by_emotion[emotion].add(dataset)

        if row.get("transcript"):
            transcript_count += 1
        if _bucket(row.get("age")) != "unknown":
            age_count += 1
        if gender != "unknown":
            gender_known_count += 1
        if row.get("arousal"):
            arousal_count += 1
        if row.get("valence"):
            valence_count += 1
        if row.get("dominance"):
            dominance_count += 1
        if row.get("speaker_id"):
            dataset_scoped_speakers.add((dataset, row["speaker_id"]))

    per_dataset_summary: list[dict[str, Any]] = []
    for dataset, dataset_rows in sorted(per_dataset_rows.items()):
        dataset_durations = [
            _safe_float(row.get("duration_seconds")) or 0.0 for row in dataset_rows if row.get("duration_seconds")
        ]
        dataset_languages = Counter(_bucket(row.get("language")) for row in dataset_rows)
        dataset_emotions = Counter(_bucket(row.get("emotion")) for row in dataset_rows)
        dataset_speakers = {
            row["speaker_id"] for row in dataset_rows if row.get("speaker_id")
        }
        duration_summary = _duration_stats(dataset_durations)

        per_dataset_summary.append(
            {
                "dataset": dataset,
                "samples": len(dataset_rows),
                "hours": round(duration_summary["total_seconds"] / 3600, 3),
                "speakers": len(dataset_speakers),
                "languages": dict(dataset_languages.most_common()),
                "top_languages": _top_labels(dataset_languages, limit=3),
                "emotions": dict(dataset_emotions.most_common()),
                "top_emotions": _top_labels(dataset_emotions, limit=5),
                "mean_seconds": duration_summary["mean_seconds"],
                "median_seconds": duration_summary["median_seconds"],
                "max_seconds": duration_summary["max_seconds"],
            }
        )

    extended_emotions = sorted(
        emotion for emotion in emotion_counter if emotion not in BASIC_EMOTIONS
    )
    extended_samples = sum(emotion_counter[emotion] for emotion in extended_emotions)

    return {
        "scope": source,
        "manifest_path": str(manifest_path),
        "overview": {
            "datasets": len(per_dataset_summary),
            "samples": total_rows,
            "hours": round(sum(duration_values) / 3600, 3),
            "dataset_scoped_speakers": len(dataset_scoped_speakers),
            "languages": len(language_counter),
            "emotion_labels": len(emotion_counter),
            "mean_clip_seconds": round(statistics.mean(duration_values), 6) if duration_values else 0.0,
            "median_clip_seconds": round(statistics.median(duration_values), 6) if duration_values else 0.0,
            "min_clip_seconds": round(min(duration_values), 6) if duration_values else 0.0,
            "max_clip_seconds": round(max(duration_values), 6) if duration_values else 0.0,
        },
        "metadata_coverage": {
            "transcript_rows": transcript_count,
            "transcript_share_percent": round((transcript_count / total_rows) * 100, 2) if total_rows else 0.0,
            "age_rows": age_count,
            "age_share_percent": round((age_count / total_rows) * 100, 2) if total_rows else 0.0,
            "gender_known_rows": gender_known_count,
            "gender_known_share_percent": round((gender_known_count / total_rows) * 100, 2) if total_rows else 0.0,
            "arousal_rows": arousal_count,
            "valence_rows": valence_count,
            "dominance_rows": dominance_count,
        },
        "per_dataset": per_dataset_summary,
        "emotion_distribution": _counter_rows(
            emotion_counter,
            total_rows,
            duration_by_emotion,
            dataset_counter_by_key=dataset_counter_by_emotion,
        ),
        "language_distribution": _counter_rows(
            language_counter,
            total_rows,
            duration_by_language,
            dataset_counter_by_key=dataset_counter_by_language,
        ),
        "gender_distribution": _counter_rows(gender_counter, total_rows, defaultdict(float)),
        "sample_rate_distribution": _counter_rows(sample_rate_counter, total_rows, defaultdict(float)),
        "channels_distribution": _counter_rows(channels_counter, total_rows, defaultdict(float)),
        "audio_format_distribution": _counter_rows(audio_format_counter, total_rows, defaultdict(float)),
        "extended_emotions": {
            "labels": extended_emotions,
            "samples": extended_samples,
            "share_percent": round((extended_samples / total_rows) * 100, 2) if total_rows else 0.0,
        },
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"


def _render_markdown(summary: dict[str, Any]) -> str:
    overview = summary["overview"]
    metadata = summary["metadata_coverage"]
    per_dataset = summary["per_dataset"]
    emotion_distribution = summary["emotion_distribution"]
    language_distribution = summary["language_distribution"]
    gender_distribution = summary["gender_distribution"]
    sample_rate_distribution = summary["sample_rate_distribution"]
    channels_distribution = summary["channels_distribution"]
    audio_format_distribution = summary["audio_format_distribution"]
    extended = summary["extended_emotions"]
    plot_artifacts = summary.get("artifacts", {}).get("plots", [])

    dataset_table = _markdown_table(
        ["Dataset", "Samples", "Hours", "Speakers", "Mean Sec", "Median Sec", "Languages", "Top Emotions"],
        [
            [
                row["dataset"],
                _format_number(row["samples"]),
                _format_number(row["hours"]),
                _format_number(row["speakers"]),
                _format_number(row["mean_seconds"]),
                _format_number(row["median_seconds"]),
                row["top_languages"],
                row["top_emotions"],
            ]
            for row in per_dataset
        ],
    )

    emotion_table = _markdown_table(
        ["Emotion", "Samples", "Share %", "Hours", "Datasets"],
        [
            [
                row["label"],
                _format_number(row["samples"]),
                _format_number(row["share_percent"]),
                _format_number(row["hours"]),
                ", ".join(row.get("datasets", [])),
            ]
            for row in emotion_distribution
        ],
    )

    language_table = _markdown_table(
        ["Language", "Samples", "Share %", "Hours", "Datasets"],
        [
            [
                row["label"],
                _format_number(row["samples"]),
                _format_number(row["share_percent"]),
                _format_number(row["hours"]),
                ", ".join(row.get("datasets", [])),
            ]
            for row in language_distribution
        ],
    )

    gender_table = _markdown_table(
        ["Gender", "Samples", "Share %"],
        [
            [row["label"], _format_number(row["samples"]), _format_number(row["share_percent"])]
            for row in gender_distribution
        ],
    )

    sample_rate_table = _markdown_table(
        ["Sample Rate Hz", "Samples", "Share %"],
        [
            [row["label"], _format_number(row["samples"]), _format_number(row["share_percent"])]
            for row in sample_rate_distribution
        ],
    )

    channels_table = _markdown_table(
        ["Channels", "Samples", "Share %"],
        [
            [row["label"], _format_number(row["samples"]), _format_number(row["share_percent"])]
            for row in channels_distribution
        ],
    )

    format_table = _markdown_table(
        ["Audio Format", "Samples", "Share %"],
        [
            [row["label"], _format_number(row["samples"]), _format_number(row["share_percent"])]
            for row in audio_format_distribution
        ],
    )

    lines = [
        "# Audio Emotion Dataset Report",
        "",
        f"- Scope: `{summary['scope']}`",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Total samples: {_format_number(overview['samples'])}",
        f"- Total hours: {_format_number(overview['hours'])}",
        f"- Dataset-scoped speakers: {_format_number(overview['dataset_scoped_speakers'])}",
        f"- Languages: {_format_number(overview['languages'])}",
        f"- Distinct emotion labels: {_format_number(overview['emotion_labels'])}",
        f"- Mean clip length: {_format_number(overview['mean_clip_seconds'])} sec",
        f"- Median clip length: {_format_number(overview['median_clip_seconds'])} sec",
        f"- Min / max clip length: {_format_number(overview['min_clip_seconds'])} sec / {_format_number(overview['max_clip_seconds'])} sec",
        "",
        "## Dataset Breakdown",
        "",
        dataset_table,
        "",
        "## Emotion Distribution",
        "",
        emotion_table,
        "",
        "## Language Distribution",
        "",
        language_table,
        "",
        "## Metadata Coverage",
        "",
        f"- Transcript coverage: {_format_number(metadata['transcript_rows'])} rows ({_format_number(metadata['transcript_share_percent'])}%)",
        f"- Age coverage: {_format_number(metadata['age_rows'])} rows ({_format_number(metadata['age_share_percent'])}%)",
        f"- Known gender coverage: {_format_number(metadata['gender_known_rows'])} rows ({_format_number(metadata['gender_known_share_percent'])}%)",
        f"- Arousal coverage: {_format_number(metadata['arousal_rows'])} rows",
        f"- Valence coverage: {_format_number(metadata['valence_rows'])} rows",
        f"- Dominance coverage: {_format_number(metadata['dominance_rows'])} rows",
        "",
        "## Gender Distribution",
        "",
        gender_table,
        "",
        "## Audio Properties",
        "",
        "### Sample Rates",
        "",
        sample_rate_table,
        "",
        "### Channels",
        "",
        channels_table,
        "",
        "### Formats",
        "",
        format_table,
        "",
        "## Notes",
        "",
        f"- Extended or non-basic emotion labels account for {_format_number(extended['samples'])} samples ({_format_number(extended['share_percent'])}%).",
        f"- Extended labels present in this scope: {', '.join(extended['labels']) if extended['labels'] else 'none'}.",
        "- `cameo` is an aggregate corpus and overlaps with some standalone sources such as `cremad` and `ravdess`.",
    ]
    if plot_artifacts:
        lines.extend(
            [
                "",
                "## Plot Files",
                "",
                *[
                    f"- `{plot['path']}`: {plot['description']}"
                    for plot in plot_artifacts
                ],
            ]
        )
    return "\n".join(lines) + "\n"


def _write_plots(
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    source: str,
    data_root: Path,
) -> list[dict[str, str]]:
    plot_dir = _plot_dir(source, data_root)
    relative_root = _report_base_dir(source, data_root)

    plot_paths = [
        (
            _save_vertical_bar_plot(
                labels=[row["dataset"] for row in summary["per_dataset"]],
                values=[row["hours"] for row in summary["per_dataset"]],
                title="Dataset Hours",
                ylabel="Hours",
                output_path=plot_dir / "dataset_hours.png",
            ),
            "Total audio hours per dataset.",
        ),
        (
            _save_horizontal_bar_plot(
                labels=[row["label"] for row in summary["emotion_distribution"]],
                values=[row["samples"] for row in summary["emotion_distribution"]],
                title="Emotion Distribution",
                xlabel="Samples",
                output_path=plot_dir / "emotion_distribution.png",
            ),
            "Sample count per emotion label.",
        ),
        (
            _save_horizontal_bar_plot(
                labels=[row["label"] for row in summary["language_distribution"]],
                values=[row["samples"] for row in summary["language_distribution"]],
                title="Language Distribution",
                xlabel="Samples",
                output_path=plot_dir / "language_distribution.png",
            ),
            "Sample count per language.",
        ),
        (
            _save_duration_histogram(
                durations=[
                    duration
                    for duration in (_safe_float(row.get("duration_seconds")) for row in rows)
                    if duration is not None
                ],
                output_path=plot_dir / "duration_histogram.png",
            ),
            "Distribution of clip durations in seconds.",
        ),
        (
            _save_dataset_emotion_heatmap(
                per_dataset=summary["per_dataset"],
                top_emotions=[row["label"] for row in summary["emotion_distribution"][:10]],
                output_path=plot_dir / "dataset_emotion_heatmap.png",
            ),
            "Dataset-by-emotion sample count heatmap for the most common labels.",
        ),
    ]

    artifacts: list[dict[str, str]] = []
    for path, description in plot_paths:
        artifacts.append(
            {
                "path": path.relative_to(relative_root).as_posix(),
                "absolute_path": str(path),
                "description": description,
            }
        )
    return artifacts


def _save_vertical_bar_plot(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    height = max(4.8, 4.0 + len(labels) * 0.2)
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.1), height))
    colors = [PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(labels))]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_horizontal_bar_plot(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    labels = list(labels)
    values = list(values)
    fig_height = max(4.5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = [PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(labels))]
    bars = ax.barh(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    max_value = max(values) if values else 0
    offset = max_value * 0.01 if max_value else 0.1
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}" if float(value).is_integer() else f"{value:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_duration_histogram(*, durations: list[float], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = max(20, min(60, len(durations) // 1200)) if durations else 20
    ax.hist(durations, bins=bins, color=PLOT_COLORS[0], edgecolor="white", alpha=0.9)
    ax.set_title("Clip Duration Histogram")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_dataset_emotion_heatmap(
    *,
    per_dataset: list[dict[str, Any]],
    top_emotions: list[str],
    output_path: Path,
) -> Path:
    datasets = [row["dataset"] for row in per_dataset]
    matrix = [
        [row["emotions"].get(emotion, 0) for emotion in top_emotions]
        for row in per_dataset
    ]

    fig_width = max(9.5, len(top_emotions) * 1.05)
    fig_height = max(4.5, len(datasets) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_title("Dataset x Emotion Heatmap")
    ax.set_xticks(range(len(top_emotions)), labels=top_emotions, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)), labels=datasets)

    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            if value:
                ax.text(
                    column_index,
                    row_index,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#0f172a",
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
