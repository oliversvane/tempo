from __future__ import annotations

import argparse

from .report import write_report
from .registry import SOURCES
from .utils import combine_manifests


def _selected_sources(source_name: str) -> list[tuple[str, object]]:
    if source_name == "all":
        return [(name, SOURCES[name]) for name in SOURCES]
    return [(source_name, SOURCES[source_name])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and preprocess audio emotion datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    choices = ["all", *SOURCES.keys()]
    for command in ("download", "preprocess", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("source", choices=choices)
        command_parser.add_argument("--force", action="store_true")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("source", nargs="?", default="all", choices=choices)

    subparsers.add_parser("list")

    args = parser.parse_args()

    if args.command == "list":
        for name in SOURCES:
            print(name)
        return 0

    if args.command == "report":
        markdown_path, json_path, plot_paths = write_report(args.source)
        print(f"Wrote Markdown report to {markdown_path}")
        print(f"Wrote JSON stats to {json_path}")
        for plot_path in plot_paths:
            print(f"Wrote plot to {plot_path}")
        return 0

    for name, module in _selected_sources(args.source):
        if args.command in {"download", "run"}:
            print(f"Downloading {name}...")
            module.download_dataset(force=args.force)
        if args.command in {"preprocess", "run"}:
            print(f"Preprocessing {name}...")
            module.preprocess_dataset(force=args.force)

    if args.command in {"preprocess", "run"}:
        combined_manifest = combine_manifests()
        print(f"Wrote combined manifest to {combined_manifest}")

    return 0
