import argparse
import sys

from tools.stats.test_dashboard import get_all_run_attempts, upload_additional_info


def select_recent_attempts(attempts: list[int], max_attempts: int) -> list[int]:
    """The newest `max_attempts` run attempts, oldest first."""
    return sorted(attempts)[-max_attempts:]


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

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    run_attempts = select_recent_attempts(
        get_all_run_attempts(args.workflow_run_id), args.max_attempts
    )

    for i in run_attempts:
        # Flush stdout so that any errors in the upload show up last in the
        # logs.
        sys.stdout.flush()
        upload_additional_info(args.workflow_run_id, i)
