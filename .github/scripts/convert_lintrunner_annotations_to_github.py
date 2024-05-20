import json
import subprocess
import sys

from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional


# From: https://docs.github.com/en/rest/reference/checks
class GitHubAnnotationLevel(str, Enum):
    NOTICE = "notice"
    WARNING = "warning"
    FAILURE = "failure"


class GitHubAnnotation(NamedTuple):
    path: str
    start_line: int
    end_line: int
    start_column: Optional[int]
    end_column: Optional[int]
    annotation_level: GitHubAnnotationLevel
    message: str
    title: Optional[str]
    raw_details: Optional[str]


PYTORCH_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    .decode("ascii")
    .strip()
)

annotations = []
for line in sys.stdin:
    lint_message = json.loads(line)

    path = lint_message.get("path")
    line = lint_message.get("line")

    code = lint_message["code"]
    severity = lint_message["severity"]
    name = lint_message["name"]
    description = lint_message.get("description")

    # These fields are required by the GitHub API, but optional in lintrunner.
    # If they don't exist, just skip.
    if path is None or line is None:
        print(f"No path/line for lint: ({code}) {name}", file=sys.stderr)
        continue

    # normalize path relative to git root
    path = Path(path).relative_to(PYTORCH_ROOT)

    annotations.append(
        GitHubAnnotation(
            path=str(path),
            start_line=int(line),
            end_line=int(line),
            start_column=None,
            end_column=None,
            annotation_level=GitHubAnnotationLevel.FAILURE,
            message=description,
            title=f"({code}) {name}",
            raw_details=None,
        )._asdict()
    )

print(json.dumps(annotations), flush=True)
