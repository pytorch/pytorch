import json
import subprocess
import sys
from enum import Enum
from typing import NamedTuple, Optional, Tuple


LINTER_CODE = "LINTRUNNER_VERSION"


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


def toVersionString(version_tuple: Tuple[int, int, int]) -> str:
    return ".".join(str(x) for x in version_tuple)


if __name__ == "__main__":
    version_str = (
        subprocess.run(["lintrunner", "-V"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    import re

    version_match = re.compile(r"lintrunner (\d+)\.(\d+)\.(\d+)").match(version_str)

    if not version_match:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="Lintrunner is not installed, did you forget to run `make setup_lint && make lint`?",
        )
        sys.exit(0)

    curr_version = int(version_match[1]), int(version_match[2]), int(version_match[3])
    min_version = (0, 10, 7)

    if curr_version < min_version:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ADVICE,
            name="command-failed",
            original=None,
            replacement=None,
            description="".join(
                (
                    f"Lintrunner is out of date (you have v{toVersionString(curr_version)} ",
                    f"instead of v{toVersionString(min_version)}). ",
                    "Please run `pip install lintrunner -U` to update it",
                )
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
