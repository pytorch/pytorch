import re
import sys
from pathlib import Path

from mypy.plugin import Plugin


def get_correct_mypy_version():
    # there's probably a more elegant way to do this
    (match,) = re.finditer(
        r"mypy==(\d+(?:\.\d+)*)",
        (
            Path(__file__).parent.parent / ".ci" / "docker" / "requirements-ci.txt"
        ).read_text(),
    )
    (version,) = match.groups()
    return version


def plugin(version: str):
    correct_version = get_correct_mypy_version()
    if version != correct_version:
        print(
            f"""\
You are using mypy version {version}, which is not supported
in the PyTorch repo. Please switch to mypy version {correct_version}.

For example, if you installed mypy via pip, run this:

    pip install mypy=={correct_version}

Or if you installed mypy via conda, run this:

    conda install -c conda-forge mypy={correct_version}
""",
            file=sys.stderr,
        )
    return Plugin
