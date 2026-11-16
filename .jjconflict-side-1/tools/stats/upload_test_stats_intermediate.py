import argparse
import sys

from tools.stats.test_dashboard import get_all_run_attempts, upload_additional_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    run_attempts = get_all_run_attempts(args.workflow_run_id)

    for i in run_attempts:
        # Flush stdout so that any errors in the upload show up last in the
        # logs.
        sys.stdout.flush()
        upload_additional_info(args.workflow_run_id, i)
