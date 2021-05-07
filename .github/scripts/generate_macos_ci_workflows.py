#!/usr/bin/env python3

import os
from pathlib import Path

import jinja2

REPO_DIR = Path(__file__).absolute().parent.parent.parent
GITHUB_DIR = REPO_DIR.joinpath(".github")
TEMPLATES_DIR = GITHUB_DIR.joinpath("templates")


def repo_path(filename):
    return filename.replace(str(REPO_DIR), "").lstrip(os.path.sep)


class MacOSWorkflow:
    def __init__(self, build_environment: str, runner_type: str, on_pull_request: bool):
        self.build_environment = build_environment

        self.args = {
            "build_environment": build_environment,
            "runner_type": runner_type,
            "on_pull_request": on_pull_request,
        }

    def generate_workflow_file(
        self, workflow_template: jinja2.Template, jinja_env: jinja2.Environment
    ) -> Path:
        output_file_path = GITHUB_DIR.joinpath(
            f"workflows/{self.build_environment}.yml"
        )
        with open(output_file_path, "w") as output_file:

            output_file.writelines(
                [
                    "# @generated DO NOT EDIT MANUALLY\n",
                    f"# Template is at:    {repo_path(workflow_template.filename)}\n",
                    f"# Generation script: {repo_path(os.path.abspath(__file__))}\n",
                ]
            )
            output_file.write(
                workflow_template.render(**self.args)
            )
            output_file.write("\n")
        return output_file_path


WORKFLOWS = [
    MacOSWorkflow(
        build_environment="pytorch-macos-10.15-py3.8",
        runner_type="macos-10.15",
        on_pull_request=True,
    ),
]


def get_jinja_env():
    return jinja2.Environment(
        # Add ! to the defualt start string so it doesn't interfere with GitHub's
        # own substitution (which is inside ${{ }} blocks)
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        undefined=jinja2.StrictUndefined,
    )


# TODO: Most of this is copy-pasted from generate_linux_ci_workflows.py, refactor
# out the common stuff
if __name__ == "__main__":
    jinja_env = get_jinja_env()

    workflow_template = jinja_env.get_template("macos_ci_workflow.yml.j2")
    for workflow in WORKFLOWS:
        output_path = workflow.generate_workflow_file(
            workflow_template=workflow_template, jinja_env=jinja_env
        )
        print(f"Generated {output_path}")
