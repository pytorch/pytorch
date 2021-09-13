from typing import List, Dict, Any
import yaml
import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Manual Target Determination
# See the source issue for discussion: https://github.com/pytorch/pytorch/issues/63781.


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

        print(
            f"Testing rule '{name}' with sources:\n{sources_info}\n"
            f"Got matches:\n{matches_info}"
        )
        print("is_match", is_match)

        if is_match:
            tests += rule["tests"]

    return tests


def determinate(diff_path: Path, rules_path: Path) -> List[str]:
    with open(rules_path) as f:
        rules = yaml.safe_load(f)

    # with open(diff_path) as f:
    #     pass

    changed_files = ["README.md", "test.md"]

    determined_tests = determine_for_files(changed_files, rules["rules"])
    core_tests = [
        "test_autograd",
        "test_modules",
        "test_nn",
        "test_ops",
        "test_torch"
    ]

    print(
        f"Running determined tests:\n{indent_list(determined_tests)}\n"
        f"With default core tests:\n{indent_list(core_tests)}"
    )

    return determined_tests + core_tests


print(determinate(Path("abc"), REPO_ROOT / "test" / "manual_determinations.yml"))
