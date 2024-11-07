import argparse
import sys

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_test_stats import get_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # Flush stdout so that any errors in Rockset upload show up last in the logs.
    sys.stdout.flush()

    upload_additional_info(args.workflow_run_id, args.workflow_run_attempt, test_cases)
