"""Parse GitHub Actions workflow YAML to extract test matrix entries."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOWS_DIR = Path(".github/workflows")


def resolve_workflow_path(name: str) -> Path:
    """
    Resolve a workflow name to a full path under .github/workflows/.
    Accepts 'pull', 'pull.yml', or a full path.
    Raises RuntimeError with available options if not found.
    """
    candidate = Path(name)
    if candidate.exists():
        return candidate

    # Try under .github/workflows/
    for suffix in ("", ".yml", ".yaml"):
        path = WORKFLOWS_DIR / f"{name}{suffix}"
        if path.exists():
            return path

    available = sorted(p.stem for p in WORKFLOWS_DIR.glob("*.y*ml") if p.is_file())
    raise RuntimeError(f"Workflow {name!r} not found. Available: {available}")


def _strip_gha(s: str) -> str:
    return re.sub(r"\$\{\{[^}]+\}\}", "", s).strip()


def _build_runner(job: dict) -> str:
    with_block = job.get("with", {})
    runner = _strip_gha(str(with_block.get("runner", "")))
    if runner:
        return runner
    # default: prefix + linux.2xlarge
    return "linux.2xlarge"


def _build_image(job: dict) -> str:
    with_block = job.get("with", {})
    return _strip_gha(str(with_block.get("docker-image-name", "")))


def parse_workflow(
    yaml_path: str,
    runner_filter: str = "linux.2xlarge",
) -> list[dict]:
    """
    Parse a GitHub Actions workflow YAML and return unique (build_env, config) entries
    whose test runner matches runner_filter.

    Each entry: {build_env, config, test_runner, build_runner}
    """
    with open(yaml_path) as f:
        workflow = yaml.safe_load(f)

    seen: set[tuple[str, str]] = set()
    results: list[dict] = []

    for job_name, job in workflow.get("jobs", {}).items():
        if not job_name.endswith("-build"):
            continue

        with_block = job.get("with", {})
        build_env = str(with_block.get("build-environment", ""))
        matrix_str = str(with_block.get("test-matrix", ""))

        if not build_env or not matrix_str:
            continue

        build_runner = _build_runner(job)
        build_image = _build_image(job)

        try:
            matrix = yaml.safe_load(_strip_gha(matrix_str))
        except yaml.YAMLError:
            continue

        for entry in matrix.get("include", []):
            config = str(entry.get("config", ""))
            test_runner = _strip_gha(str(entry.get("runner", "")))

            if runner_filter not in test_runner:
                continue

            key = (build_env, config)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "build_env": build_env,
                    "config": config,
                    "test_runner": test_runner,
                    "build_runner": build_runner,
                    "build_image": build_image,
                }
            )

    return results
