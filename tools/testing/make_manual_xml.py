import os
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_timestamp() -> str:
    return datetime.now().isoformat()


def sanitize_sysout(sysout: str, quotes: bool = True) -> str:
    # Source: https://stackoverflow.com/questions/1546717/escaping-strings-for-use-in-xml
    trans_dict = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
    }
    if quotes:
        trans_dict.update(
            {
                "'": "&apos;",
                '"': "&quot;",
            }
        )
    table = str.maketrans(trans_dict)
    return sysout.translate(table)


def make_manual_xml(
    invoking_file: str, pytest_testname: str, duration: float, sysout: str
) -> None:
    """Generate a JUnit XML string for a single test run."""

    sanitized_invoking_file = invoking_file.replace("/", ".")

    xml_path = (
        REPO_ROOT
        / "test"
        / "test-reports"
        / "python-pytest"
        / sanitized_invoking_file
        / f"{sanitized_invoking_file}-{os.urandom(8).hex()}.xml"
    )
    re_pat = r"^(?P<file>.*)::(?P<classname>.*)::(?P<testname>.*)$"
    re_match = re.match(re_pat, pytest_testname)
    assert re_match, f"Failed to match {pytest_testname} with {re_pat}"
    file = re_match.group("file")
    classname = re_match.group("classname")
    testname = re_match.group("testname")
    testcase = (
        f'<testcase name="{testname}" classname="{classname}" file="{file}" time="{duration}">'
        f'<failure message="{sanitize_sysout(sysout)}">{sanitize_sysout(sysout, quotes=False)}'
        f"</failure></testcase>"
    )
    testsuite = (
        f'<testsuite name="{classname}" tests="1" errors="0" failures="1" skipped="0" '
        f'time="{duration}" timestamp="{get_timestamp()}">{testcase}</testsuite>'
    )
    s = f"<testsuites>{testsuite}</testsuites>"

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xml_path, "w") as f:
        f.write(s)
