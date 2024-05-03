from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List
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

keyword_synonyms = [
    ["amp", "mixed_precision"],
    ["quantized", "quantization", "quant", "quantize"],
    ["decomp", "decomposition", "decompositions"],
    ["numpy", "torch_np", "numpy_tests"],
    ["ops", "opinfo"],
]

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

custom_matchers: Dict[str, Callable[[str], bool]] = {
    "nn": lambda x: "nn" in x.replace("onnx", "_"),
    "c10": lambda x: "c10" in x.replace("c10d", "_"),
}


def get_keywords(file: str) -> List[str]:
    keywords = []
    for folder in Path(file).parts[:-1]:
        folder = sanitize_folder_name(folder)
        keywords.append(folder)

    return [kw for kw in keywords if kw not in not_keyword]


def sanitize_folder_name(folder_name: str) -> str:
    if folder_name.startswith("_"):
        folder_name = folder_name[1:]

    for syns in keyword_synonyms:
        if folder_name in syns:
            return syns[0]

    return folder_name


class FilePath(HeuristicInterface):
    # Heuristic based on folders in the file path.  Takes each folder of each
    # changed file and attempts to find matches based on those folders
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        keyword_frequency: Dict[str, int] = defaultdict(int)
        try:
            changed_files = query_changed_files()
        except Exception as e:
            warn(f"Can't query changed test files due to {e}")
            changed_files = []

        for cf in changed_files:
            keywords = get_keywords(cf)
            for keyword in keywords:
                keyword_frequency[keyword] += 1

        test_ratings: Dict[str, float] = defaultdict(float)

        for test in tests:
            for keyword, frequency in keyword_frequency.items():
                if custom_matchers.get(keyword, lambda x: keyword in x)(str(test)):  # type: ignore[no-untyped-call]
                    test_ratings[test] += frequency
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
