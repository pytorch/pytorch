#!/usr/bin/env python3
"""Map EC2 runner labels to ARC equivalents using .github/arc.yaml.

Takes a GitHub Actions test matrix, replaces each runner with its ARC
equivalent, and prints the updated matrix as JSON.

Usage:
    python map_ec2_to_arc.py --prefix mt- '{ include: [
      { config: "default", shard: 1, num_shards: 5, runner: "mt-linux.4xlarge" },
    ]}'
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map EC2 runner labels to ARC runner labels in a test matrix"
    )
    parser.add_argument(
        "matrix",
        help="GitHub Actions test matrix string to transform",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Runner prefix to strip from labels (e.g. 'mt-')",
    )
    return parser.parse_args()


def strip_prefix(label: str, prefix: str) -> str:
    if prefix and label.startswith(prefix):
        return label[len(prefix) :]
    return label


def load_mapping(arc_yaml: Path) -> dict[str, str]:
    with open(arc_yaml) as f:
        data = yaml.safe_load(f)
    return data["runner_mapping"]


def set_output(name: str, val: str) -> None:
    print(f"Setting {name}={val}")
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            print(f"{name}={val}", file=f)


def main() -> None:
    args = parse_args()
    arc_yaml = Path(__file__).resolve().parent.parent / "arc.yaml"
    mapping = load_mapping(arc_yaml)

    matrix = yaml.safe_load(args.matrix)
    if not matrix:
        set_output("test-matrix", args.matrix)
        return

    entries = matrix.get("include", [])
    if not entries:
        set_output("test-matrix", json.dumps(matrix))
        return

    # TODO(huydo): onnxruntime uses hardware_concurrency() to size its thread
    # pool, which sees all host CPUs (e.g., 192) on ARC k8s instead of the
    # container's cpuset (e.g., 16). This causes pthread_setaffinity_np errors.
    # Skip onnx tests on ARC until the onnxruntime session options are fixed to
    # use cgroup-aware CPU counts.
    excluded_configs = {"onnx"}
    filtered = []
    for entry in entries:
        if entry.get("config") in excluded_configs:
            print(f"Excluding config '{entry['config']}' from ARC test matrix")
            continue
        filtered.append(entry)
    matrix["include"] = filtered

    for entry in filtered:
        if "runner" not in entry:
            continue
        clean = strip_prefix(entry["runner"].strip(), args.prefix)
        if clean not in mapping:
            print(f"error: no ARC runner found for '{clean}'", file=sys.stderr)
            sys.exit(1)
        entry["runner"] = args.prefix + mapping[clean]

    set_output("test-matrix", json.dumps(matrix))


if __name__ == "__main__":
    main()
