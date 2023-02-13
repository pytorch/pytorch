#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from typing_extensions import TypedDict  # Python 3.11+

Step = Dict[str, Any]


class Script(TypedDict):
    extension: str
    script: str


def extract(step: Step) -> Optional[Script]:
    run = step.get("run")

    # https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#using-a-specific-shell
    shell = step.get("shell", "bash")
    extension = {
        "bash": ".sh",
        "pwsh": ".ps1",
        "python": ".py",
        "sh": ".sh",
        "cmd": ".cmd",
        "powershell": ".ps1",
    }.get(shell)

    is_gh_script = step.get("uses", "").startswith("actions/github-script@")
    gh_script = step.get("with", {}).get("script")

    if run is not None and extension is not None:
        script = {
            "bash": f"#!/usr/bin/env bash\nset -eo pipefail\n{run}",
            "sh": f"#!/usr/bin/env sh\nset -e\n{run}",
        }.get(shell, run)
        return {"extension": extension, "script": script}
    elif is_gh_script and gh_script is not None:
        return {"extension": ".js", "script": gh_script}
    else:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; aborting to avoid overwriting")

    gha_expressions_found = False

    for p in Path(".github/workflows").iterdir():
        with open(p, "rb") as f:
            workflow = yaml.safe_load(f)

        for job_name, job in workflow["jobs"].items():
            job_dir = out / p / job_name
            if "steps" not in job:
                continue
            steps = job["steps"]
            index_chars = len(str(len(steps) - 1))
            for i, step in enumerate(steps, start=1):
                extracted = extract(step)
                if extracted:
                    script = extracted["script"]
                    step_name = step.get("name", "")
                    if "${{" in script:
                        gha_expressions_found = True
                        print(
                            f"{p} job `{job_name}` step {i}: {step_name}",
                            file=sys.stderr,
                        )

                    job_dir.mkdir(parents=True, exist_ok=True)

                    sanitized = re.sub(
                        "[^a-zA-Z_]+",
                        "_",
                        f"_{step_name}",
                    ).rstrip("_")
                    extension = extracted["extension"]
                    filename = f"{i:0{index_chars}}{sanitized}{extension}"
                    (job_dir / filename).write_text(script)

    if gha_expressions_found:
        sys.exit(
            "Each of the above scripts contains a GitHub Actions "
            "${{ <expression> }} which must be replaced with an `env` variable"
            " for security reasons."
        )


if __name__ == "__main__":
    main()
