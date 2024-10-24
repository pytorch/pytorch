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

import difflib
import sys
from io import StringIO
from pathlib import Path

import ruamel.yaml  # type: ignore[import]


def fn(base: str) -> str:
    return str(base / Path("aten/src/ATen/native/native_functions.yaml"))


with open(Path(__file__).parent.parent.parent / fn(".")) as f:
    contents = f.read()

yaml = ruamel.yaml.YAML(typ='safe')
yaml.preserve_quotes = True  # type: ignore[assignment]
yaml.width = 1000  # type: ignore[assignment]
yaml.boolean_representation = ["False", "True"]  # type: ignore[attr-defined]
r = yaml.load(contents)

# Cuz ruamel's author intentionally didn't include conversion to string
# https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-to-string-not-to-stream
string_stream = StringIO()
yaml.dump(r, string_stream)
new_contents = string_stream.getvalue()
string_stream.close()

if contents != new_contents:
    print(
        """\

## LINT FAILURE: native_functions.yaml ##

native_functions.yaml failed lint; please apply the diff below to fix lint.
If you think this is in error, please see .github/scripts/lint_native_functions.py
""",
        file=sys.stderr,
    )
    sys.stdout.writelines(
        difflib.unified_diff(
            contents.splitlines(True), new_contents.splitlines(True), fn("a"), fn("b")
        )
    )
    sys.exit(1)
