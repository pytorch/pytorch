import yaml
import textwrap
import os
import subprocess
import argparse

from typing import Dict, List, Any

REPO_ROOT = os.path.dirname(os.path.dirname(__name__))
CONFIG_YML = os.path.join(REPO_ROOT, ".circleci", "config.yml")
WORKFLOWS_DIR = os.path.join(REPO_ROOT, ".github", "workflows")

WORKFLOWS_TO_CHECK = [
    "binary_builds",
    "build",
    "master_build",
    # These are formatted slightly differently, skip them
    # "scheduled-ci",
    # "debuggable-scheduled-ci",
    # "slow-gradcheck-scheduled-ci",
    # "ecr_gc",
    # "promote",
]


def add_job(
    workflows: Dict[str, Any],
    workflow_name: str,
    type: str,
    job: Dict[str, Any],
    past_jobs: Dict[str, Any],
) -> None:
    """
    Add job 'job' under 'type' and 'workflow_name' to 'workflow' in place. Also
    add any dependencies (they must already be in 'past_jobs')
    """
    if workflow_name not in workflows:
        workflows[workflow_name] = {"when": "always", "jobs": []}

    requires = job.get("requires", None)
    if requires is not None:
        for reuqirement in requires:
            dependency = past_jobs[reuqirement]
            add_job(workflows, dependency["workflow_name"], dependency["type"], dependency["job"], past_jobs)

    workflows[workflow_name]["jobs"].append({type: job})


def get_filtered_circleci_config(
    workflows: Dict[str, Any], relevant_jobs: List[str]
) -> Dict[str, Any]:
    """
    Given an existing CircleCI config, remove every job that's not listed in
    'relevant_jobs'
    """
    new_workflows: Dict[str, Any] = {}
    past_jobs: Dict[str, Any] = {}
    for workflow_name, workflow in workflows.items():
        if workflow_name not in WORKFLOWS_TO_CHECK:
            # Don't care about this workflow, skip it entirely
            continue

        for job_dict in workflow["jobs"]:
            for type, job in job_dict.items():
                if "name" not in job:
                    # Job doesn't have a name so it can't be handled
                    print("Skipping", type)
                else:
                    if job["name"] in relevant_jobs:
                        # Found a job that was specified at the CLI, add it to
                        # the new result
                        add_job(new_workflows, workflow_name, type, job, past_jobs)

                    # Record the job in case it's needed as a dependency later
                    past_jobs[job["name"]] = {
                        "workflow_name": workflow_name,
                        "type": type,
                        "job": job,
                    }

    return new_workflows


def commit_ci(files: List[str], message: str):
    # Check that there are no other modified files than the ones edited by this
    # tool
    stdout = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE).stdout.decode()
    for line in stdout.split("\n"):
        if line == "":
            continue
        if line[0] != " ":
            raise RuntimeError(f"Refusing to commit while other changes are already staged: {line}")


    # Make the commit
    subprocess.run(["git", "add", CONFIG_YML, ".github/workflows/."])
    subprocess.run(["git", "commit", "-m", message])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="make .circleci/config.yml only have a specific set of jobs and delete GitHub actions"
    )
    parser.add_argument("--job", action="append", help="job name", required=True)
    parser.add_argument(
        "--keep-gha", action="store_true", help="don't delete GitHub actions"
    )
    parser.add_argument(
        "--make-commit", action="store_true", help="add change to git with to a do-not-merge commit"
    )
    args = parser.parse_args()

    touched_files = [CONFIG_YML]
    config_yml = yaml.safe_load(open(CONFIG_YML, "r").read())
    config_yml["workflows"] = get_filtered_circleci_config(config_yml["workflows"], args.job)
    yaml.dump(config_yml, open(CONFIG_YML, "w"))

    if not args.keep_gha:
        for f in os.listdir(WORKFLOWS_DIR):
            path = os.path.join(WORKFLOWS_DIR, f)
            touched_files.append(path)
            os.remove(path)

    if args.make_commit:
        jobs_str = '\n'.join([f" * {job}" for job in args.job])
        message = textwrap.dedent(f"""
        [skip ci][do not merge] Edit config.yml to filter specific jobs

        Filter CircleCI to only run:
        {jobs_str}

        See [Run Specific CI Jobs](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#run-specific-ci-jobs) for details.
        """).strip()
        commit_ci(touched_files, message)
