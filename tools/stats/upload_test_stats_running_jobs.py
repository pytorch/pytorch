import sys
import time
from functools import cache
from typing import Any

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import get_s3_resource


BUCKET_PREFIX = "workflows_failing_pending_upload"


@cache
def get_bucket() -> Any:
    return get_s3_resource().Bucket("gha-artifacts")


def delete_obj(key: str) -> None:
    # Does not raise error if key does not exist
    get_bucket().delete_objects(
        Delete={
            "Objects": [{"Key": key}],
            "Quiet": True,
        }
    )


def put_object(key: str) -> None:
    get_bucket().put_object(
        Key=key,
        Body=b"",
    )


def do_upload(workflow_id: int) -> None:
    workflow_attempt = 1
    # Flush stdout so that any errors in upload show up last in the logs.
    sys.stdout.flush()
    upload_additional_info(workflow_id, workflow_attempt)


def get_workflow_ids(pending: bool = False) -> list[int]:
    prefix = f"{BUCKET_PREFIX}/{'pending/' if pending else ''}"
    objs = get_bucket().objects.filter(Prefix=prefix)
    return [int(obj.key.split("/")[-1].split(".")[0]) for obj in objs]


def read_s3(pending: bool = False) -> None:
    while True:
        workflows = get_workflow_ids(pending)
        if not workflows:
            if pending:
                break
            # Wait for more stuff to show up
            print("Sleeping for 60 seconds")
            time.sleep(60)
        for workflow_id in workflows:
            print(f"Processing {workflow_id}")
            put_object(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")
            delete_obj(f"{BUCKET_PREFIX}/{workflow_id}.txt")
            try:
                do_upload(workflow_id)
            except Exception as e:
                print(f"Failed to upload {workflow_id}: {e}")
            delete_obj(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")


if __name__ == "__main__":
    # Workflows in the pending folder were previously in progress of uploading
    # but failed to complete, so we need to retry them.
    read_s3(pending=True)
    read_s3()
