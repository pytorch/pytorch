#!/usr/bin/env python3

# these two pages have some relevant information:
# https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
# https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

Job = Dict[str, Any]

windows_labels = {'windows-latest', 'windows-2019'}


def get_default_shell(job: Job) -> str:
    return 'pwsh' if job['runs-on'] in windows_labels else 'bash'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f'{out} already exists; aborting to avoid overwriting')

    gha_expressions_found = False

    for p in Path('.github/workflows').iterdir():
        with open(p) as f:
            workflow = yaml.safe_load(f)

        for job_name, job in workflow['jobs'].items():
            job_dir = out / p / job_name
            default_shell = get_default_shell(job)
            steps = job['steps']
            index_chars = len(str(len(steps) - 1))
            for i, step in enumerate(steps, start=1):
                script = step.get('run')
                if script:
                    step_name = step['name']
                    if '${{' in script:
                        gha_expressions_found = True
                        print(
                            f'{p} job `{job_name}` step {i}: {step_name}',
                            file=sys.stderr
                        )

                    if step.get('shell', default_shell) == 'bash':
                        job_dir.mkdir(parents=True, exist_ok=True)

                        sanitized = re.sub(
                            '[^a-zA-Z_]+', '_',
                            f'_{step_name}',
                        ).rstrip('_')
                        filename = f'{i:0{index_chars}}{sanitized}.sh'
                        (job_dir / filename).write_text(
                            f'#!/usr/bin/env bash\nset -eo pipefail\n{script}'
                        )

    if gha_expressions_found:
        sys.exit(
            'Each of the above scripts contains a GitHub Actions '
            '${{ <expression> }} which must be replaced with an `env` variable'
            ' for security reasons.'
        )


if __name__ == '__main__':
    main()
