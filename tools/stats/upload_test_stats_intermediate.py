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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="only upload the most recent N run attempts",
    )
    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    run_attempts = sorted(get_all_run_attempts(args.workflow_run_id))
    run_attempts = run_attempts[-args.max_attempts : -1]

    for i in run_attempts:
        # Flush stdout so that any errors in the upload show up last in the
        # logs.
        sys.stdout.flush()
        try:
            upload_additional_info(args.workflow_run_id, i)
        except Exception:
            pass
