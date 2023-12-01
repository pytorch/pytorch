import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))
from tools.stats.import_test_stats import (
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)


def main() -> None:
    print("Exporting files from test-infra")
    get_test_times()
    get_test_class_times()
    get_test_file_ratings()
    get_test_class_ratings()
    get_td_heuristic_historial_edited_files_json()
    get_td_heuristic_profiling_json()
    copy_pytest_cache()


if __name__ == "__main__":
    main()
