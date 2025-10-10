from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
)
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).parents[3]

keyword_synonyms: dict[str, list[str]] = {
    "amp": ["mixed_precision"],
    "quant": ["quantized", "quantization", "quantize"],
    "decomp": ["decomposition", "decompositions"],
    "numpy": ["torch_np", "numpy_tests"],
    "ops": ["opinfo"],
    "hop": ["higher_order_op"],
    "aot": ["flex_attention", "autograd"],
    "inductor": ["dynamo", "export"],  # not actually synonyms but they interact a lot
}


custom_matchers: dict[str, Callable[[str], bool]] = {
    "nn": lambda x: "nn" in x.replace("onnx", "_"),
    "c10": lambda x: "c10" in x.replace("c10d", "_"),
}


def is_valid_keyword(keyword: str) -> bool:
    not_keyword = [
        "torch",
        "test",
        "tests",
        "util",
        "utils",
        "func",
        "src",
        "c",
        "ns",
        "tools",
        "internal",
    ]
    return keyword == "nn" or (keyword not in not_keyword and len(keyword) > 2)


@lru_cache(maxsize=1)
def get_keywords(file: str) -> list[str]:
    keywords = []
    for folder in Path(file).parts[:-1]:
        folder = sanitize_name(folder)
        keywords.append(folder)

    file_name = Path(file).stem.split("_")
    keywords.extend([sanitize_name(x) for x in file_name])
    return [kw for kw in keywords if is_valid_keyword(kw)]


def sanitize_name(folder_name: str) -> str:
    folder_name = folder_name.removeprefix("_")

    for syn_rep, syns in keyword_synonyms.items():
        if folder_name in syns or folder_name == syn_rep:
            return syn_rep

    return folder_name


def file_matches_keyword(file: str, keyword: str) -> bool:
    keywords = get_keywords(file)
    return (
        keyword in keywords
        or any(
            syn in keywords or syn in file for syn in keyword_synonyms.get(keyword, [])
        )
        or custom_matchers.get(keyword, lambda x: keyword in x)(file)  # type: ignore[no-untyped-call]
    )


def get_freq_dict(tests: list[str], changed_files: list[str]) -> dict[str, int]:
    keyword_frequency: dict[str, int] = defaultdict(int)
    for cf in changed_files:
        keywords = get_keywords(cf)
        for keyword in keywords:
            keyword_frequency[keyword] += 1

    test_ratings: dict[str, int] = defaultdict(int)

    for test in tests:
        for keyword, frequency in keyword_frequency.items():
            if file_matches_keyword(test, keyword):
                test_ratings[test] += frequency
    return test_ratings


class Filepath(HeuristicInterface):
    # Heuristic based on folders in the file path.  Takes each folder of each
    # changed file and attempts to find matches based on those folders
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        try:
            changed_files = query_changed_files()
        except Exception as e:
            warn(f"Can't query changed test files due to {e}")
            changed_files = []

        test_ratings = get_freq_dict(tests, changed_files)
        test_ratings = {
            TestRun(k): float(v) for (k, v) in test_ratings.items() if k in tests
        }
        return TestPrioritizations(
            tests, normalize_ratings(test_ratings, 0.25, min_value=0.125)
        )


if __name__ == "__main__":
    # Quick thing so you can call the heuristic from the command line with a sha
    import os
    import sys

    from tools.testing.discover_tests import TESTS

    git_diff = f"git diff --name-only {sys.argv[1]} {sys.argv[1]}^"
    changed_files = os.popen(git_diff).read().split("\n")
    freq_dict = get_freq_dict(
        TESTS, [x for x in changed_files if x != "" and not x.startswith("test")]
    )
    for k, v in sorted(freq_dict.items(), key=lambda x: x[1], reverse=False):
        print(k, v)
    print(changed_files)
