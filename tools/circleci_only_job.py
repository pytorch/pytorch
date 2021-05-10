import yaml
import os
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(__name__))
CONFIG_YML = os.path.join(REPO_ROOT, ".circleci", "config.yml")
WORKFLOWS_DIR = os.path.join(REPO_ROOT, ".github", "workflows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make config.yml only have a specific set of jobs and delete GitHub actions")
    parser.add_argument("--job", action="append", help="job name", required=True)
    args = parser.parse_args()

    config_yml = yaml.safe_load(open(CONFIG_YML, "r").read())

    workflows = dict(config_yml["workflows"])

    relevant_jobs = args.job

    workflows_to_check = [
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


    new_workflows = {}

    def add_job(workflow_name, type, job):
        if workflow_name not in new_workflows:
            new_workflows[workflow_name] = {
                "when": "always",
                "jobs": []
            }

        requires = job.get("requires", None)
        if requires is not None:
            for req in requires:
                add_job(**past_jobs[req])


        new_workflows[workflow_name]["jobs"].append({
            type: job
        })

    past_jobs = {}

    workflow_items = list(workflows.items())
    for workflow_name, workflow in workflows.items():
        if workflow_name not in workflows_to_check:
            continue
        for job_dict in workflow["jobs"]:
            for type, job in job_dict.items():
                if "name" not in job:
                    print("Skipping", type)
                else:
                    if job["name"] in relevant_jobs:
                        add_job(workflow_name, type, job)
                    past_jobs[job["name"]] = {
                        "workflow_name": workflow_name,
                        "type": type,
                        "job": job
                    }


    config_yml["workflows"] = new_workflows


    yaml.dump(config_yml, open(CONFIG_YML, "w"))

    for f in os.listdir(WORKFLOWS_DIR):
        os.remove(os.path.join(WORKFLOWS_DIR, f))
