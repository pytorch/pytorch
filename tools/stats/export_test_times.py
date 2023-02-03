import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))
from tools.stats.import_test_stats import get_test_times

TEST_TIMES_FILE = ".pytorch-test-times.json"


def main() -> None:
    print(f"Exporting test times from test-infra to {TEST_TIMES_FILE}")
    get_test_times(str(REPO_ROOT), filename=TEST_TIMES_FILE)


if __name__ == "__main__":
    main()
