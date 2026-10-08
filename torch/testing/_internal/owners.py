import json
import re
import subprocess
from pathlib import Path


OWNERS_PREFIXES = ("# Owner(s): ", "// Owner(s): ")
IGNORED_OWNERS = ["unknown", "tests"]
TEST_GLOBS = [
    # Python tests.
    "test/**/test_*.py",
    "test/**/*_test.py",
    "test/distributed/**/*_example.py",
    # Binary validation tests.
    ".ci/pytorch/test_example_code/**/*.cpp",
    # Native tests.
    "test/**/*.cpp",
    "test/**/*.cc",
    "test/**/*.cxx",
    "test/**/*.cu",
    "test/**/*.h",
    "test/**/*.mm",
    # Android tests.
    "android/pytorch_android/src/androidTest/cpp/**/*.cpp",
    # ATen tests.
    "aten/src/ATen/**/test/**/*.cpp",
    "aten/src/ATen/**/test/**/*.cc",
    "aten/src/ATen/**/test/**/*.cu",
    "aten/src/ATen/**/test/**/*.h",
    "aten/src/ATen/**/test/**/*.mm",
    # MPS tests.
    "aten/src/ATen/native/metal/mpscnn/tests/**/*.h",
    "aten/src/ATen/native/metal/mpscnn/tests/**/*.mm",
    # Static runtime tests.
    "benchmarks/static_runtime/**/*.cc",
    "benchmarks/static_runtime/**/*.h",
    # c10 tests.
    "c10/**/test/**/*.cpp",
    "c10/**/test/**/*.cu",
    "c10/**/test/**/*.h",
    # Tests alongside implementation code.
    "aten/src/ATen/core/**/*_test.cpp",
    "caffe2/**/*_test.cc",
]


def load_codeowners_rules(
    codeowners_path: Path,
) -> list[tuple[re.Pattern[str], str | None]]:
    if not codeowners_path.is_file():
        return []

    rules = []
    review_area = None
    for line in codeowners_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# [") and line.endswith("]"):
            review_area = line[3:-1].strip() or None
        elif line.startswith(("# /", "/")):
            # Commented paths also declare review areas, without assigning reviewers.
            pattern = line.removeprefix("# ").split()[0].strip("/")
            expression = re.escape(pattern)
            expression = expression.replace(r"\*\*/", "(?:.*/)?")
            expression = expression.replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
            rules.append((re.compile(expression + "(?:/.*)?"), review_area))
    return rules


def collect_owners(repo_root: Path) -> dict[str, list[str]]:
    codeowners_rules = load_codeowners_rules(repo_root / "CODEOWNERS")
    test_paths = {
        test_path
        for test_glob in TEST_GLOBS
        for test_path in repo_root.glob(test_glob)
        if test_path.is_file()
    }
    owners_by_file = {}
    for test_path in sorted(test_paths):
        relative_path = test_path.relative_to(repo_root)
        if (
            relative_path.as_posix() == "test/run_test.py"
            or "fb" in relative_path.parts
            or "third_party" in relative_path.parts
        ):
            continue
        owner_labels = []
        with test_path.open(encoding="utf-8") as test_file:
            for line in test_file:
                if line.startswith(OWNERS_PREFIXES):
                    try:
                        owner_labels = json.loads(line.partition(": ")[2])
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid owners header in {relative_path}"
                        ) from error
                    if not isinstance(owner_labels, list) or any(
                        not isinstance(label, str) for label in owner_labels
                    ):
                        raise ValueError(
                            f"Expected a list of owner labels in {relative_path}"
                        )
                    break
        file_owners = {
            label.removeprefix("module: ")
            for label in owner_labels
            if label.startswith("module: ")
        }.difference(IGNORED_OWNERS)
        if not file_owners:
            for pattern, review_area in reversed(codeowners_rules):
                if pattern.fullmatch(relative_path.as_posix()):
                    if review_area is not None:
                        file_owners = {review_area}
                    break
        owners_by_file[relative_path.as_posix()] = sorted(
            file_owners.difference(IGNORED_OWNERS)
        )
    if not owners_by_file:
        raise ValueError(f"No test files found in {repo_root}")
    return owners_by_file


if __name__ == "__main__":
    print(
        json.dumps(
            collect_owners(
                repo_root=Path(
                    subprocess.check_output(
                        ["git", "rev-parse", "--show-toplevel"], text=True
                    ).strip()
                )
            ),
            indent=2,
        )
    )
