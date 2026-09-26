#!/usr/bin/env python3
"""Map EC2 runner labels to ARC equivalents using .github/arc.yaml.

Takes a GitHub Actions test matrix, replaces each runner with its ARC
equivalent, and prints the updated matrix as JSON.

Usage:
    python map_ec2_to_arc.py --prefix mt- '{ include: [
      { config: "default", shard: 1, num_shards: 5, runner: "mt-linux.4xlarge" },
    ]}'

LF allowlist (pytorch/ci-infra#1081): with prefix "lf-", only ARC runners in
    the allowlist keep it; others are force-routed to "mt-". The allowlist
    comes from --lf-runners if given, else arc.yaml's `lf_allowlist`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


LF_PREFIX = "lf-"
META_PREFIX = "mt-"
LF_MODES = ("all", "restricted")


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
    parser.add_argument(
        "--lf-runners",
        default=None,
        help=(
            "Comma-separated ARC runners allowed on LF; overrides arc.yaml's "
            "lf_allowlist. Empty string means unrestricted."
        ),
    )
    return parser.parse_args()


def strip_prefix(label: str, prefix: str) -> str:
    if prefix and label.startswith(prefix):
        return label[len(prefix) :]
    return label


def load_arc_yaml(arc_yaml: Path) -> dict:
    with open(arc_yaml) as f:
        return yaml.safe_load(f)


def load_mapping(data: dict) -> dict[str, str]:
    return data["runner_mapping"]


def load_meta_only(data: dict) -> set[str]:
    return set(data.get("meta_only_runners") or [])


def parse_lf_runners_arg(value: str) -> frozenset[str] | None:
    """Parse --lf-runners. Empty string -> None (unrestricted)."""
    runners = frozenset(r.strip() for r in value.split(",") if r.strip())
    return runners or None


def load_lf_config(data: dict) -> tuple[str, frozenset[str]]:
    """Validate and extract arc.yaml's lf_allowlist (ci-infra#1081).

    Always validates mode, even if the caller ends up using --lf-runners
    instead, so a broken arc.yaml isn't masked by an override.
    """
    allowlist = data.get("lf_allowlist") or {}
    mode = allowlist.get("mode", "all")
    if mode not in LF_MODES:
        print(
            f"error: lf_allowlist.mode must be one of {LF_MODES}, got '{mode}'",
            file=sys.stderr,
        )
        sys.exit(1)
    runners = frozenset(allowlist.get("runners") or [])
    return mode, runners


def resolve_lf_allowlist(
    lf_runners_arg: str | None, data: dict
) -> frozenset[str] | None:
    """--lf-runners wins when given (mode is still validated); else arc.yaml.

    An empty runners: list under mode: restricted also means unrestricted,
    matching what an explicit --lf-runners "" means -- otherwise the same
    arc.yaml would mean two different things depending on which of the two
    code paths reads it (ci-infra#1081).
    """
    mode, runners = load_lf_config(data)
    if lf_runners_arg is not None:
        return parse_lf_runners_arg(lf_runners_arg)
    if mode == "restricted":
        return runners or None
    return None


def get_arc_yaml_path() -> Path:
    # Overridable so tests can point at a fixture arc.yaml without touching
    # the real config.
    override = os.environ.get("ARC_YAML_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "arc.yaml"


def set_output(name: str, val: str) -> None:
    print(f"Setting {name}={val}")
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            print(f"{name}={val}", file=f)


def main() -> None:
    args = parse_args()
    arc_yaml = get_arc_yaml_path()
    data = load_arc_yaml(arc_yaml)
    mapping = load_mapping(data)
    meta_only = load_meta_only(data)
    lf_allowlist = resolve_lf_allowlist(args.lf_runners, data)
    if lf_allowlist is not None:
        print(f"LF allowlist active ({len(lf_allowlist)} runners)")

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
        mapped = mapping[clean]
        # Some hardware (H100, B200) exists only on the Meta fleet, so force the
        # 'mt-' prefix regardless of the build job's fleet assignment.
        if clean in meta_only:
            entry["runner"] = META_PREFIX + mapped
            continue
        # Passthrough runners (e.g. linux.rocm.gpu.2, linux.idc.xpu) are not
        # OSDC-managed so they keep their original label without the prefix.
        if mapped == clean:
            entry["runner"] = mapped
            continue
        # Active allowlist: only listed ARC targets may stay on LF; else
        # force-route to Meta, same as meta_only_runners (ci-infra#1081).
        if (
            lf_allowlist is not None
            and args.prefix == LF_PREFIX
            and mapped not in lf_allowlist
        ):
            print(f"'{clean}' -> '{mapped}' not in LF allowlist; forcing {META_PREFIX}")
            entry["runner"] = META_PREFIX + mapped
            continue
        entry["runner"] = args.prefix + mapped

    set_output("test-matrix", json.dumps(matrix))


if __name__ == "__main__":
    main()
