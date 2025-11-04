import re
from pathlib import Path
from typing import Any


HAS_PYYAML = True
try:
    import yaml
except ImportError:
    print("Please install pyyaml to use target determination features.")
    HAS_PYYAML = False


REPO_ROOT = Path(__file__).resolve().parents[3]


def get_job_info_from_workflow_file(workflow_file: str) -> list[list[dict[str, Any]]]:
    """
    Returns groups of jobs that are similar based on the test configurations
    they run.

    This is pretty hardcoded, so it is fragile, but it returns a pretty accurate
    mapping

    TODO replace with better (automated?) system. Maybe a separate workflow that
    generates an artifact that says which jobs are similar according what tests
    they run, correlation etc, also looks at jobs on main branch or merge base
    to better determine what jobs exist.
    """
    if not HAS_PYYAML:
        return []
    # Usually takes the form
    # pytorch/pytorch/.github/workflows/pull.yml@refs/pull/165793/merge in CI?
    workflow_file = workflow_file.split("@")[0].split(".github/workflows/")
    workflow_file = ".github/workflows/" + workflow_file[1]

    regex = r"needs\.([a-zA-Z0-9_-]+)\.outputs\.test-matrix"

    with open(REPO_ROOT / workflow_file) as f:
        yml = yaml.safe_load(f)
    raw_jobs = yml.get("jobs", {})
    jobs: list[dict[str, Any]] = []
    dependent_jobs = {}

    for job, job_info in raw_jobs.items():
        if "test-matrix" not in job_info.get("with", {}):
            continue
        try:
            test_matrix = yaml.safe_load(job_info["with"]["test-matrix"])
            if "include" not in test_matrix:
                # ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm86-build.outputs.test-matrix }}
                match = re.search(regex, test_matrix)
                if match:
                    dep_job = match.group(1)
                    dependent_jobs[f"{job_info.get('name', job)}"] = {
                        "depends_on": f"{dep_job}",
                        "uses": job_info.get("uses", ""),
                    }
                continue
        except yaml.YAMLError as e:
            print(f"Error parsing test-matrix for job {job}: {e}")
            continue
        jobs.append(
            {
                "job_id": f"{job}",
                "job_name": f"{job_info.get('name', job)}",
                "test_matrix": sorted(
                    {entry["config"] for entry in test_matrix["include"]}
                ),
                "uses": job_info.get("uses", ""),
            }
        )

    # Fill in dependent jobs
    for job, info in dependent_jobs.items():
        for j in jobs:
            if j["job_id"] == info["depends_on"]:
                jobs.append(
                    {
                        "job_id": job,
                        "job_name": job,
                        "test_matrix": j["test_matrix"],
                        "uses": info["uses"],
                    }
                )
                break

    # Remove everything that doesn't use test
    jobs = [j for j in jobs if "test" in j["uses"]]

    individual_jobs = [
        {"job_name": j["job_name"], "config": config, "uses": j.get("uses", "")}
        for j in jobs
        for config in j["test_matrix"]
    ]

    # Group the jobs together
    # generally same test config -> same group
    grouped_jobs: dict[str, list[dict[str, Any]]] = {}
    for job in individual_jobs:
        key = []
        if "onnx" in job["job_name"]:
            key.append("onnx")
        if "bazel" in job["job_name"]:
            key.append("bazel")
        if "cuda" in job["job_name"]:
            key.append("cuda")
        if "mac" in job["job_name"]:
            key.append("mac")
        if "windows" in job["job_name"]:
            key.append("windows")
        key.append(job["config"])
        key.append(job["uses"])
        key_str = "|".join(sorted(key))
        if key_str not in grouped_jobs:
            grouped_jobs[key_str] = []
        grouped_jobs[key_str].append(job)

    for group in grouped_jobs.values():
        for j in group:
            j.pop("uses", None)

    return list(grouped_jobs.values())
