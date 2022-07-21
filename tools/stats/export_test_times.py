import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_TIMES_FILE = ".pytorch-test-times.json"

try:
    sys.path.append(str(REPO_ROOT))
    from tools.stats.import_test_stats import get_test_times

    def main() -> None:
        print(f"Exporting test times from test-infra to {TEST_TIMES_FILE}")
        get_test_times(str(REPO_ROOT), filename=TEST_TIMES_FILE)

    if __name__ == "__main__":
        main()
except ImportError:
    print("failed to import")

# the JSON file to store the S3 test stats
