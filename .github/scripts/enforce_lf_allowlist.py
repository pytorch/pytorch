#!/usr/bin/env python3
"""Enforce the LF allowlist on a GitHub Actions test matrix.

OSDC's pod-naming migration (pytorch/pytorch#198700, #198703, #198705,
#198706) made every workflow name its literal ARC pod directly, retiring
map_ec2_to_arc.py's EC2->ARC translation. The one piece of that script
which the migration doesn't replace -- forcing a test-matrix entry off
the LF fleet when its ARC target isn't in the allowlist -- has no other
home, so it's re-hosted here (ci-infra#1081).

Usage:
    python enforce_lf_allowlist.py --lf-runners "l-x86iavx512-8-64" '{ include: [
      { config: "default", shard: 1, num_shards: 5, runner: "lf-l-x86iavx512-8-64" },
    ]}'
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

# TODO(huydo): onnxruntime uses hardware_concurrency() to size its thread
# pool, which sees all host CPUs (e.g., 192) on ARC k8s instead of the
# container's cpuset (e.g., 16). This causes pthread_setaffinity_np errors.
# Skip onnx tests on ARC until the onnxruntime session options are fixed to
# use cgroup-aware CPU counts.
EXCLUDED_CONFIGS = {"onnx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force non-allowlisted LF runners in a test matrix onto the Meta fleet"
    )
    parser.add_argument(
        "matrix",
        help="GitHub Actions test matrix string to transform",
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


def load_arc_yaml(arc_yaml: Path) -> dict:
    with open(arc_yaml) as f:
        return yaml.safe_load(f)


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
    data = load_arc_yaml(get_arc_yaml_path())
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

    filtered = []
    for entry in entries:
        if entry.get("config") in EXCLUDED_CONFIGS:
            print(f"Excluding config '{entry['config']}' from ARC test matrix")
            continue
        filtered.append(entry)
    matrix["include"] = filtered

    for entry in filtered:
        raw = entry.get("runner", "").strip()
        # Callers now name their ARC pod directly, so an entry not currently
        # on 'lf-' (already 'mt-', or some other passthrough label) needs no
        # enforcement -- checking the entry's own literal prefix, rather than
        # this job's runner_prefix input, is what correctly leaves alone a
        # matrix that hardcodes 'mt-' on specific entries (e.g. H100/B200)
        # while the job itself builds on a dynamically-resolved 'lf-'.
        if lf_allowlist is None or not raw.startswith(LF_PREFIX):
            continue
        clean = raw[len(LF_PREFIX) :]
        if clean not in lf_allowlist:
            print(f"'{clean}' not in LF allowlist; forcing {META_PREFIX}")
            entry["runner"] = META_PREFIX + clean

    set_output("test-matrix", json.dumps(matrix))


if __name__ == "__main__":
    main()
