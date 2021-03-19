#!/usr/bin/env python3

from pathlib import Path
from typing import List

import jinja2
from typing_extensions import TypedDict


class PyTorchWindowsWorkflow(TypedDict):
    build_environment: str
    cuda_version: str
    name: str
    use_cuda: str


GITHUB_DIR = Path(__file__).parent.parent


def generate_workflow_file(
    *,
    workflow: PyTorchWindowsWorkflow,
    workflow_template: jinja2.Template,
) -> Path:
    output_file_path = GITHUB_DIR / f'workflows/{workflow["build_environment"]}.yml'
    with open(output_file_path, 'w') as output_file:
        output_file.write(workflow_template.render(**workflow))
        output_file.write('\n')
    return output_file_path


WORKFLOWS: List[PyTorchWindowsWorkflow] = [
    # {
    #     'build_environment': 'pytorch-win-vs2019-cuda10-cudnn7-py3',
    #     'cuda_version': '10.1',
    #     'name': 'pytorch_windows_vs2019_py36_cuda10.1_build',
    #     'use_cuda': '1',
    # },
    # {
    #     'build_environment': 'pytorch-win-vs2019-cuda11-cudnn8-py3',
    #     'cuda_version': '11.1',
    #     'name': 'pytorch_windows_vs2019_py36_cuda11.1_build',
    #     'use_cuda': '1',
    # },
    {
        'build_environment': 'pytorch-win-vs2019-cpu-py3',
        'cuda_version': 'cpu',
        'name': 'pytorch_windows_vs2019_py36_cpu_build',
        'use_cuda': '0',
    },
]


def main() -> None:
    jinja_env = jinja2.Environment(
        variable_start_string='!{{',
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR / 'templates')),
    )
    workflow_template = jinja_env.get_template('windows_ci_workflow.yml.in')
    for workflow in WORKFLOWS:
        print(generate_workflow_file(
            workflow=workflow,
            workflow_template=workflow_template,
        ))


if __name__ == '__main__':
    main()
