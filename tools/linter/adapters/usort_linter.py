import argparse
import concurrent.futures
import json
import os
import subprocess
from enum import Enum
from typing import List, NamedTuple, Optional

from usort import config as usort_config, usort


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


IS_WINDOWS: bool = os.name == "nt"


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def check_file(
    filename: str,
) -> List[LintMessage]:
    try:
        top_of_file_cat = usort_config.Category("top_of_file")
        known = usort_config.known_factory()
        # cinder magic imports must be on top (after future imports)
        known["__strict__"] = top_of_file_cat
        known["__static__"] = top_of_file_cat

        config = usort_config.Config(
            categories=(
                (
                    usort_config.CAT_FUTURE,
                    top_of_file_cat,
                    usort_config.CAT_STANDARD_LIBRARY,
                    usort_config.CAT_THIRD_PARTY,
                    usort_config.CAT_FIRST_PARTY,
                )
            ),
            known=known,
        )

        with open(filename, mode="rb") as f:
            original = f.read()
            result = usort(original, config)
            if result.error:
                raise result.error

    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="USORT",
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=(
                    "usort timed out while trying to process a file. "
                    "Please report an issue in pytorch/torchrec."
                ),
            )
        ]
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="USORT",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
        ]

    replacement = result.output
    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            line=None,
            char=None,
            code="USORT",
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="Run `lintrunner -a` to apply this patch.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with usort.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(check_file, filename): filename
            for filename in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                raise RuntimeError(f"Failed at {futures[future]}")


if __name__ == "__main__":
    main()
