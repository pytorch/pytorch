import argparse

from pytest_caching_utils import *

def main():
    parser = argparse.ArgumentParser(description="Upload this job's the pytest cache to S3")
    parser.add_argument('--pr_identifier', required=True, help='A unique PR identifier')
    parser.add_argument('--workflow', required=True, help='The workflow name')
    parser.add_argument('--job', required=True, help='The job name')
    parser.add_argument('--shard', required=True, help='The shard id')
    parser.add_argument('--bucket', required=False, help='The S3 bucket to upload the cache to')
    args = parser.parse_args()

    # The command to run this script is:
    # python .github/scripts/pytest_upload_cache.py --pr_identifier ${{ github.event.inputs.pr_identifier }} --workflow_name ${{ github.workflow }} --job_name ${{ github.job }} --shard_id ${{ github.run_number }} --bucket ${{ secrets.S3_BUCKET }}
 
    print(args)
    #download_pytest_cache(args.pr_identifier, args.bucket, args.job_name, args.temp_dir, args.pytest_cache_dir_new)

if __name__ == '__main__':
    main()