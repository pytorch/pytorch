import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.stats.import_test_stats import get_test_class_times, get_test_times


def main() -> None:
    print("Exporting test times from test-infra")
    get_test_times()
    get_test_class_times()


if __name__ == "__main__":
    main()
