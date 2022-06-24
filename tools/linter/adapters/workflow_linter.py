"""Linter to enforce invariants on our core CI workflows."""
import argparse
import json
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from yaml import CSafeLoader, dump, load

CODE = "WORKFLOW"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


class Workflow:
    def __init__(self, path: str):
        self.path = Path(path)
        with open(self.path) as f:
            self.yaml = load(f, CSafeLoader)

    def jobs(self) -> List["Job"]:
        return [Job(self, job_id) for job_id in self.yaml["jobs"]]


class Job:
    def __init__(self, workflow: Workflow, id: str):
        self.workflow = workflow
        self.id = id
        self.yaml = workflow.yaml["jobs"][id]


def get_line(path: Path, job_id: str) -> int:
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if f"{job_id}:" in line:
            return i + 1
    raise ValueError(f"Could not find line for job {job_id} in {path}")


def print_lint_message(path: Path, job_id: str, name: str, description: str) -> None:
    lint_message = LintMessage(
        path=str(path),
        line=get_line(path, job_id),
        char=None,
        code=CODE,
        severity=LintSeverity.ERROR,
        name=name,
        original=None,
        replacement=None,
        description=description,
    )
    print(json.dumps(lint_message._asdict()), flush=True)


def do_sync_tags_check(workflows: List[Workflow]) -> None:
    """Check that all jobs wth a sync-tag have the same source code as all other
    jobs with a matching sync-tag.
    """
    # Go through the provided files, aggregating jobs with the same sync tag
    tag_to_jobs = defaultdict(list)

    for workflow in workflows:
        for job in workflow.jobs():
            try:
                sync_tag = job.yaml["with"]["sync-tag"]
            except KeyError:
                continue

            tag_to_jobs[sync_tag].append(job)

    # For each sync tag, check that all the jobs have the same code.
    for sync_tag, jobs in tag_to_jobs.items():
        baseline_job = jobs.pop()

        def dump_no_if(job: Job) -> Any:
            # remove the "if" field, which we allow to be different between jobs
            # (since you might have different triggering conditions on pull vs.
            # trunk, say.)
            # deepcopy first to avoid mutating the job object
            job_yaml = deepcopy(job.yaml)
            if "if" in job_yaml:
                del job_yaml["if"]
            return dump(job_yaml)

        baseline_str = dump_no_if(baseline_job)

        printed_baseline = False
        for job in jobs:
            job_str = dump_no_if(job)
            if baseline_str != job_str:
                print_lint_message(
                    job.workflow.path,
                    job.id,
                    "sync-tag",
                    f"Job doesn't match other jobs with sync-tag: '{sync_tag}'. "
                    "The job definition must be identical, except for if-conditions.",
                )

                if not printed_baseline:
                    print_lint_message(
                        baseline_job.workflow.path,
                        baseline_job.id,
                        "sync-tag",
                        f"Job doesn't match other jobs with sync-tag: '{sync_tag}'"
                        "The job definition must be identical, except for if-conditions.",
                    )
                    printed_baseline = True


def do_build_artifact_uniqueness_check(workflows: List[Workflow]) -> None:
    """Check that all jobs that upload build artifacts produce a unique one."""
    for workflow in workflows:
        artifact_names: Dict[str, Job] = {}
        for job in workflow.jobs():
            if "_linux-build.yml" not in job.yaml.get("uses"):
                continue

            artifact_name = job.yaml["with"]["artifact-name"]
            if artifact_name == "":
                continue

            if artifact_name in artifact_names:
                print_lint_message(
                    workflow.path,
                    job.id,
                    "build-artifact-collision",
                    "This job uploads a build artifact with the same name as:\n"
                    f"  '{artifact_names[artifact_name].id}'",
                )
            artifact_names[artifact_name] = job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="workflow linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    workflows = [Workflow(path) for path in args.filenames]
    do_sync_tags_check(workflows)
    do_build_artifact_uniqueness_check(workflows)
