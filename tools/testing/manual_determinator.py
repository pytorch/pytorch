import yaml
import fnmatch
import sys
import re

from typing import List, Dict, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Manual Target Determination
# See the source issue for discussion: https://github.com/pytorch/pytorch/issues/63781.
def sprint(s: str) -> None:
    lines = s.split("\n")
    for line in lines:
        print(f"[manual_determinator] {line}", file=sys.stderr)


def any_glob_matches(name: str, globs: List[str]) -> bool:
    for glob in globs:
        if fnmatch.fnmatch(name, glob):
            return True
    return False


def indent_list(items: List[str]) -> str:
    if len(items) == 0:
        return "  <no items>"
    items = [f"  {item}" for item in items]
    return "\n".join(items)


def filenames_from_diff(diff_path: Path) -> List[str]:
    FILENAME_RE = re.compile(r"^diff --git a\/(.*) b\/(.*)$")
    with open(diff_path) as f:
        lines = f.readlines()

    files = set()
    for line in lines:
        match = FILENAME_RE.match(line.strip())
        if match is not None:
            groups = match.groups()
            files.add(groups[0])
            files.add(groups[1])

    return list(files)


Rule = Dict[str, Any]


def determine_for_files(changed_files: List[str], rules: List[Rule]) -> List[str]:
    """
    For a git diff, generate a list of tests to run based on the changed files an
    the provided rules.

    changed_files: List of filenames relative to the repo root
    rules: Rules object from manual_determinations.yml
    """
    tests = []
    for rule in rules:
        name = rule["name"]
        sources = rule["sources"]
        matches = [any_glob_matches(filename, sources) for filename in changed_files]
        is_match = all(matches)
        matches_info = indent_list(
            [f"{filename}: {match}" for filename, match in zip(changed_files, matches)]
        )
        sources_info = indent_list(sources)

        sprint(
            f"Testing rule '{name}' with sources:\n{sources_info}\n"
            f"Got matches:\n{matches_info}"
        )
        sprint(f"is_match: {is_match}")

        if is_match:
            tests += rule["tests"]

    return tests


def nonexistent_tests(rules: List[Rule], all_tests: List[str]) -> List[str]:
    bad_tests = []

    rule_tests = set()
    for rule in rules:
        for test in rule["tests"]:
            rule_tests.add(test)

    for test in rule_tests:
        if test not in all_tests:
            bad_tests.append(test)

    return bad_tests


def determinate(
    diff_path: Path, rules_path: Path, all_tests: List[str], core_tests: List[str]
) -> List[str]:
    print("opening")
    with open(rules_path) as f:
        rules = yaml.safe_load(f)["rules"]

    print("checking non tests")
    bad_tests = nonexistent_tests(rules, all_tests)
    if len(bad_tests) > 0:
        raise RuntimeError(
            f"These tests were specified in '{rules_path}' but "
            f"are not present in the tests used for determination:\n{indent_list(bad_tests)}"
        )

    print("filenames_from_diff")
    changed_files = filenames_from_diff(diff_path)
    # changed_files = ["README.md", "test.cpp"]

    print("determine_for_files")
    determined_tests = determine_for_files(changed_files, rules)

    sprint(
        f"Running determined tests:\n{indent_list(determined_tests)}\n"
        f"With default core tests:\n{indent_list(core_tests)}"
    )

    return determined_tests + core_tests


if __name__ == "__main__":
    print(
        determinate(
            REPO_ROOT / "test.diff",
            REPO_ROOT / "test" / "manual_determinations.yml",
            [],
            ["abc"],
        )
    )
