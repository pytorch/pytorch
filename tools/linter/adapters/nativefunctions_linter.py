#!/usr/bin/env python3
"""
Verify that it is possible to round-trip native_functions.yaml via ruamel under some
configuration.  Keeping native_functions.yaml consistent in this way allows us to
run codemods on the file using ruamel without introducing line noise.  Note that we don't
want to normalize the YAML file, as that would to lots of spurious lint failures.  Anything
that ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!

ruamel is a bit picky about inconsistent indentation, so you will have to indent your
file properly.  Also, if you are working on changing the syntax of native_functions.yaml,
you may find that you want to use some format that is not what ruamel prefers.  If so,
it is OK to modify this script (instead of reformatting native_functions.yaml)--the point
is simply to make sure that there is *some* configuration of ruamel that can round trip
the YAML, not to be prescriptive about it.
"""

import argparse
import json
import sys
from enum import Enum
from io import StringIO
from typing import NamedTuple, Optional

import ruamel.yaml  # type: ignore[import]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--native-functions-yml",
        required=True,
        help="location of native_functions.yaml",
    )

    args = parser.parse_args()

    with open(args.native_functions_yml) as f:
        contents = f.read()

    yaml = ruamel.yaml.YAML()  # type: ignore[attr-defined]
    yaml.preserve_quotes = True  # type: ignore[assignment]
    yaml.width = 1000  # type: ignore[assignment]
    yaml.boolean_representation = ["False", "True"]  # type: ignore[attr-defined]
    try:
        r = yaml.load(contents)
    except Exception as err:
        msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code="NATIVEFUNCTIONS",
            severity=LintSeverity.ERROR,
            name="YAML load failure",
            original=None,
            replacement=None,
            description=f"Failed due to {err.__class__.__name__}:\n{err}",
        )

        print(json.dumps(msg._asdict()), flush=True)
        sys.exit(0)

    # Cuz ruamel's author intentionally didn't include conversion to string
    # https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-to-string-not-to-stream
    string_stream = StringIO()
    yaml.dump(r, string_stream)
    new_contents = string_stream.getvalue()
    string_stream.close()

    if contents != new_contents:
        msg = LintMessage(
            path=args.native_functions_yml,
            line=None,
            char=None,
            code="NATIVEFUNCTIONS",
            severity=LintSeverity.ERROR,
            name="roundtrip inconsistency",
            original=contents,
            replacement=new_contents,
            description=(
                "YAML roundtrip failed; run `lintrunner --take NATIVEFUNCTIONS -a` to apply the suggested changes. "
                "If you think this is in error, please see tools/linter/adapters/nativefunctions_linter.py"
            ),
        )

        print(json.dumps(msg._asdict()), flush=True)
