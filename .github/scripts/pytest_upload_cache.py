import argparse

from pytest_caching_utils import *

def main():
    parser = argparse.ArgumentParser(description="Upload this job's the pytest cache to S3")
    parser.add_argument('--cache_dir', required=True, help='Path to the folder containing the pytest cache')
    parser.add_argument('--pr_identifier', required=True, help='A unique PR identifier')
    parser.add_argument('--workflow', required=True, help='The workflow name')
    parser.add_argument('--job', required=True, help='The job name')
    parser.add_argument('--shard', required=True, help='The shard id')
    parser.add_argument('--temp_dir', required=False, help='Directory to store temp files in')
    parser.add_argument('--bucket', required=False, help='The S3 bucket to upload the cache to')
    args = parser.parse_args()
    print(args)

    # TODO: First check if it's even worth uploading a new cache: 
    #    Does the cache even mark any failed tests?

    upload_pytest_cache(
        pr_identifier=args.pr_identifier, 
        workflow=args.workflow, 
        job=args.job, 
        shard=args.shard, 
        pytest_cache_dir=args.pytest_cache_dir,
        bucket=args.bucket,
        temp_dir=args.temp_dir,
    )


if __name__ == '__main__':
    main()