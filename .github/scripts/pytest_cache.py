import argparse
import sys
from pathlib import Path

from pytest_caching_utils import (
    download_pytest_cache,
    GithubRepo,
    PRIdentifier,
    upload_pytest_cache,
)

TEMP_DIR = "./tmp"  # a backup location in case one isn't provided


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload this job's the pytest cache to S3"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--upload", action="store_true", help="Upload the pytest cache to S3"
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="Download the pytest cache from S3, merging it with any local cache",
    )

    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Path to the folder pytest uses for its cache",
    )
    parser.add_argument("--pr_identifier", required=True, help="A unique PR identifier")
    parser.add_argument(
        "--job_identifier",
        required=True,
        help="A unique job identifier that should be the same for all runs of job",
    )
    parser.add_argument(
        "--sha", required="--upload" in sys.argv, help="SHA of the commit"
    )  # Only required for upload
    parser.add_argument(
        "--test_config", required="--upload" in sys.argv, help="The test config"
    )  # Only required for upload
    parser.add_argument(
        "--shard", required="--upload" in sys.argv, help="The shard id"
    )  # Only required for upload

    parser.add_argument(
        "--repo",
        required=False,
        help="The github repository we're running in, in the format 'owner/repo-name'",
    )
    parser.add_argument(
        "--temp_dir", required=False, help="Directory to store temp files"
    )
    parser.add_argument(
        "--bucket", required=False, help="The S3 bucket to upload the cache to"
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    pr_identifier = PRIdentifier(args.pr_identifier)
    print(f"PR identifier for `{args.pr_identifier}` is `{pr_identifier}`")

    repo = GithubRepo.from_string(args.repo)
    cache_dir = Path(args.cache_dir)
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = Path(TEMP_DIR)

    if args.upload:
        print(f"Uploading cache with args {args}")

        # verify the cache dir exists
        if not cache_dir.exists():
            print(f"The pytest cache dir `{cache_dir}` does not exist. Skipping upload")
            return

        upload_pytest_cache(
            pr_identifier=pr_identifier,
            repo=repo,
            job_identifier=args.job_identifier,
            sha=args.sha,
            test_config=args.test_config,
            shard=args.shard,
            cache_dir=cache_dir,
            bucket=args.bucket,
            temp_dir=temp_dir,
        )

    if args.download:
        print(f"Downloading cache with args {args}")
        download_pytest_cache(
            pr_identifier=pr_identifier,
            repo=repo,
            job_identifier=args.job_identifier,
            dest_cache_dir=cache_dir,
            bucket=args.bucket,
            temp_dir=temp_dir,
        )


if __name__ == "__main__":
    main()
