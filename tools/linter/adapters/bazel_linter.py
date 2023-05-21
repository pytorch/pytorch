"""
This linter ensures that users don't set a SHA hash checksum in Bazel for the http_archive.
Although the security practice of setting the checksum is good, it doesn't work when the
archive is downloaded from some sites like GitHub because it can change. Specifically,
GitHub gives no guarantee to keep the same value forever. Check for more details at
https://github.com/community/community/discussions/46034.
"""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, NamedTuple, Optional, Set
from urllib.parse import urlparse


LINTER_CODE = "BAZEL_LINTER"
SHA256_REGEX = re.compile(r"\s*sha256\s*=\s*['\"](?P<sha256>[a-zA-Z0-9]{64})['\"]\s*,")
DOMAINS_WITH_UNSTABLE_CHECKSUM = {"github.com"}


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


def is_required_checksum(urls: List[Optional[str]]) -> bool:
    if not urls:
        return False

    for url in urls:
        if not url:
            continue

        parsed_url = urlparse(url)
        if parsed_url.hostname in DOMAINS_WITH_UNSTABLE_CHECKSUM:
            return False

    return True


def get_disallowed_checksums(
    binary: str,
) -> Set[str]:
    """
    Return the set of disallowed checksums from all http_archive rules
    """
    # Use bazel to get the list of external dependencies in XML format
    proc = subprocess.run(
        [binary, "query", "kind(http_archive, //external:*)", "--output=xml"],
        capture_output=True,
    )

    stdout = str(proc.stdout, "utf-8").strip()
    root = ET.fromstring(stdout)

    disallowed_checksums = set()
    # Parse all the http_archive rules in the XML output
    for rule in root.findall('.//rule[@class="http_archive"]'):
        urls_node = rule.find('.//list[@name="urls"]')
        if urls_node is None:
            continue
        urls = [n.get("value") for n in urls_node.findall(".//string")]

        checksum_node = rule.find('.//string[@name="sha256"]')
        if checksum_node is None:
            continue
        checksum = checksum_node.get("value")

        if not checksum:
            continue

        if not is_required_checksum(urls):
            disallowed_checksums.add(checksum)

    return disallowed_checksums


def check_bazel(
    filename: str,
    disallowed_checksums: Set[str],
) -> List[LintMessage]:
    original = ""
    replacement = ""

    with open(filename) as f:
        for line in f:
            original += f"{line}"

            m = SHA256_REGEX.match(line)
            if m:
                sha256 = m.group("sha256")

                if sha256 in disallowed_checksums:
                    continue

            replacement += f"{line}"

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ADVICE,
                name="format",
                original=original,
                replacement=replacement,
                description="Found redundant SHA checksums. Run `lintrunner -a` to apply this patch.",
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A custom linter to detect redundant SHA checksums in Bazel",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="bazel binary path",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    try:
        disallowed_checksums = get_disallowed_checksums(args.binary)
    except Exception as e:
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(f"Failed due to {e.__class__.__name__}:\n{e}"),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    for filename in args.filenames:
        for lint_message in check_bazel(filename, disallowed_checksums):
            print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
