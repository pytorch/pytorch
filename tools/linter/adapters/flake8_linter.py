import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Pattern, Set


IS_WINDOWS: bool = os.name == "nt"


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


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


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# fmt: off
# https://www.flake8rules.com/
DOCUMENTED_IN_FLAKE8RULES: Set[str] = {
    "E101", "E111", "E112", "E113", "E114", "E115", "E116", "E117",
    "E121", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "E129",
    "E131", "E133",
    "E201", "E202", "E203",
    "E211",
    "E221", "E222", "E223", "E224", "E225", "E226", "E227", "E228",
    "E231",
    "E241", "E242",
    "E251",
    "E261", "E262", "E265", "E266",
    "E271", "E272", "E273", "E274", "E275",
    "E301", "E302", "E303", "E304", "E305", "E306",
    "E401", "E402",
    "E501", "E502",
    "E701", "E702", "E703", "E704",
    "E711", "E712", "E713", "E714",
    "E721", "E722",
    "E731",
    "E741", "E742", "E743",
    "E901", "E902", "E999",
    "W191",
    "W291", "W292", "W293",
    "W391",
    "W503", "W504",
    "W601", "W602", "W603", "W604", "W605",
    "F401", "F402", "F403", "F404", "F405",
    "F811", "F812",
    "F821", "F822", "F823",
    "F831",
    "F841",
    "F901",
    "C901",
}

# https://pypi.org/project/flake8-comprehensions/#rules
DOCUMENTED_IN_FLAKE8COMPREHENSIONS: Set[str] = {
    "C400", "C401", "C402", "C403", "C404", "C405", "C406", "C407", "C408", "C409",
    "C410",
    "C411", "C412", "C413", "C414", "C415", "C416",
}

# https://github.com/PyCQA/flake8-bugbear#list-of-warnings
DOCUMENTED_IN_BUGBEAR: Set[str] = {
    "B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008", "B009", "B010",
    "B011", "B012", "B013", "B014", "B015",
    "B301", "B302", "B303", "B304", "B305", "B306",
    "B901", "B902", "B903", "B950",
}
# fmt: on


# stdin:2: W802 undefined name 'foo'
# stdin:3:6: T484 Name 'foo' is not defined
# stdin:3:-100: W605 invalid escape sequence '\/'
# stdin:3:1: E302 expected 2 blank lines, found 1
RESULTS_RE: Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<code>\S+?):?
    \s(?P<message>.*)
    $
    """
)


def _test_results_re() -> None:
    """
    >>> def t(s): return RESULTS_RE.search(s).groupdict()

    >>> t(r"file.py:80:1: E302 expected 2 blank lines, found 1")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '80', 'column': '1', 'code': 'E302',
     'message': 'expected 2 blank lines, found 1'}

    >>> t(r"file.py:7:1: P201: Resource `stdout` is acquired but not always released.")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '7', 'column': '1', 'code': 'P201',
     'message': 'Resource `stdout` is acquired but not always released.'}

    >>> t(r"file.py:8:-10: W605 invalid escape sequence '/'")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '8', 'column': '-10', 'code': 'W605',
     'message': "invalid escape sequence '/'"}
    """
    pass


def _run_command(
    args: List[str],
    *,
    extra_env: Optional[Dict[str, str]],
) -> "subprocess.CompletedProcess[str]":
    logging.debug(
        "$ %s",
        " ".join(
            ([f"{k}={v}" for (k, v) in extra_env.items()] if extra_env else []) + args
        ),
    )
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
            check=True,
            encoding="utf-8",
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: List[str],
    *,
    extra_env: Optional[Dict[str, str]],
    retries: int,
) -> "subprocess.CompletedProcess[str]":
    remaining_retries = retries
    while True:
        try:
            return _run_command(args, extra_env=extra_env)
        except subprocess.CalledProcessError as err:
            if remaining_retries == 0 or not re.match(
                r"^ERROR:1:1: X000 linting with .+ timed out after \d+ seconds",
                err.stdout,
            ):
                raise err
            remaining_retries -= 1
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )
            time.sleep(1)


def get_issue_severity(code: str) -> LintSeverity:
    # "B901": `return x` inside a generator
    # "B902": Invalid first argument to a method
    # "B903": __slots__ efficiency
    # "B950": Line too long
    # "C4": Flake8 Comprehensions
    # "C9": Cyclomatic complexity
    # "E2": PEP8 horizontal whitespace "errors"
    # "E3": PEP8 blank line "errors"
    # "E5": PEP8 line length "errors"
    # "F401": Name imported but unused
    # "F403": Star imports used
    # "F405": Name possibly from star imports
    # "T400": type checking Notes
    # "T49": internal type checker errors or unmatched messages
    if any(
        code.startswith(x)
        for x in [
            "B9",
            "C4",
            "C9",
            "E2",
            "E3",
            "E5",
            "F401",
            "F403",
            "F405",
            "T400",
            "T49",
        ]
    ):
        return LintSeverity.ADVICE

    # "F821": Undefined name
    # "E999": syntax error
    if any(code.startswith(x) for x in ["F821", "E999"]):
        return LintSeverity.ERROR

    # "F": PyFlakes Error
    # "B": flake8-bugbear Error
    # "E": PEP8 "Error"
    # "W": PEP8 Warning
    # possibly other plugins...
    return LintSeverity.WARNING


def get_issue_documentation_url(code: str) -> str:
    if code in DOCUMENTED_IN_FLAKE8RULES:
        return f"https://www.flake8rules.com/rules/{code}.html"

    if code in DOCUMENTED_IN_FLAKE8COMPREHENSIONS:
        return "https://pypi.org/project/flake8-comprehensions/#rules"

    if code in DOCUMENTED_IN_BUGBEAR:
        return "https://github.com/PyCQA/flake8-bugbear#list-of-warnings"

    return ""


def check_files(
    filenames: List[str],
    flake8_plugins_path: Optional[str],
    severities: Dict[str, LintSeverity],
    retries: int,
) -> List[LintMessage]:
    try:
        proc = run_command(
            [sys.executable, "-mflake8", "--exit-zero"] + filenames,
            extra_env={"FLAKE8_PLUGINS_PATH": flake8_plugins_path}
            if flake8_plugins_path
            else None,
            retries=retries,
        )
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="FLAKE8",
                severity=LintSeverity.ERROR,
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
                        stderr=err.stderr.strip() or "(empty)",
                        stdout=err.stdout.strip() or "(empty)",
                    )
                ),
            )
        ]

    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=f"{match['message']}\nSee {get_issue_documentation_url(match['code'])}",
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code="FLAKE8",
            severity=severities.get(match["code"]) or get_issue_severity(match["code"]),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(proc.stdout)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flake8 wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--flake8-plugins-path",
        help="FLAKE8_PLUGINS_PATH env value",
    )
    parser.add_argument(
        "--severity",
        action="append",
        help="map code to severity (e.g. `B950:advice`)",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out flake8",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    flake8_plugins_path = (
        None
        if args.flake8_plugins_path is None
        else os.path.realpath(args.flake8_plugins_path)
    )

    severities: Dict[str, LintSeverity] = {}
    if args.severity:
        for severity in args.severity:
            parts = severity.split(":", 1)
            assert len(parts) == 2, f"invalid severity `{severity}`"
            severities[parts[0]] = LintSeverity(parts[1])

    lint_messages = check_files(
        args.filenames, flake8_plugins_path, severities, args.retries
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
