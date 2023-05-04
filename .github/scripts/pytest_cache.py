import argparse
import sys
from pytest_caching_utils import *

def main():
    parser = argparse.ArgumentParser(description="Upload this job's the pytest cache to S3")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--upload', action='store_true', help='Upload the pytest cache to S3')
    mode.add_argument('--download', action='store_true', help='Download the pytest cache from S3, merging it with any local cache')

    parser.add_argument('--cache_dir', required=True, help='Path to the folder pytest uses for its cache')
    parser.add_argument('--pr_identifier', required=True, help='A unique PR identifier')
    parser.add_argument('--workflow', required=True, help='The workflow name')
    parser.add_argument('--job', required=True, help='The job name')
    parser.add_argument('--shard', required='--upload' in sys.argv, help='The shard id') # Only required for upload

    parser.add_argument('--temp_dir', required=False, help='Directory to store temp files')
    parser.add_argument('--bucket', required=False, help='The S3 bucket to upload the cache to')

    args = parser.parse_args()

    if args.upload:
        print(f"Uploading cache with args {args}")

        # TODO: First check if it's even worth uploading a new cache:
        #    Does the cache even mark any failed tests?

        # verify the cache dir exists
        if not os.path.exists(args.cache_dir):
            # raise an invalid input exception
            raise ValueError(f"The given pytest cache dir `{args.cache_dir}` does not exist")

        print (os.getenv("AWS_ACCESS_KEY_ID"))
        upload_pytest_cache(
            pr_identifier=PRIdentifier(args.pr_identifier),
            workflow=args.workflow,
            job=args.job,
            shard=args.shard,
            cache_dir=args.cache_dir,
            bucket=args.bucket,
            temp_dir=args.temp_dir,
        )

    if args.download:
        print(f"Downloading cache with args {args}")
        download_pytest_cache(
            pr_identifier=PRIdentifier(args.pr_identifier),
            workflow=args.workflow,
            job=args.job,
            dest_cache_dir=args.cache_dir,
            bucket=args.bucket,
            temp_dir=args.temp_dir,
        )


if __name__ == '__main__':
    main()