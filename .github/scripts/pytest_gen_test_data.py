import os
import shutil
import contextlib

from s3_upload_utils import *

PYTEST_CACHE_KEY_PREFIX = "pytest_cache"
PYTEST_CACHE_DIR_NAME = ".pytest_cache"
BUCKET = "pytest_cache"
TEMP_DIR = "/tmp" # a backup location in case one isn't provided


def create_test_files_in_folder(pytest_cache_dir):
    import random
    import os

    # delete anything currently in pytest_cache_dir
    if os.path.exists(pytest_cache_dir):
        shutil.rmtree(pytest_cache_dir)

    # make sure folder exists
    subdir = f"{pytest_cache_dir}/v/cache/"
    ensure_dir_exists(subdir)

    failed_tests_file = os.path.join(subdir, "lastfailed")
    failed_tests = {}

    # random integer from 100 to 300
    data_id = random.randint(100, 300)

    for test_num in range(10): # number of tests to fail
        test_name = f"test_id_{data_id}__failes_num_{test_num}"
        failed_tests[test_name] = True

    write_json_file(failed_tests_file, failed_tests)
    print(f"Created file {failed_tests_file}")

    other_cache_files = [
        "v/cache/file1",
        "v/cache/file2",
        "v/cache/file3",
        "v/cache/file4",
        "randofile1",
        "randofile2",
        "randofile3",
        "randofile4",
        "randofile5",
    ]

    # These are all files we assume are irrelevant to us, but we'll copy them over if they exist
    for file_name in other_cache_files:
        # Don't generate all the files. Only generate ~half of them
        if random.randint(0, len(other_cache_files)) >= (len(other_cache_files) // 2):
            continue

        file_path = os.path.join(pytest_cache_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(f"This is a test file from generateion {data_id}")
        print("Created file {}".format(file_path))

if __name__ == '__main__':
    folder = f"/Users/zainr/deleteme/test-files"
    subfolder = f"{folder}/fake_pytest_cache"
    create_test_files_in_folder(subfolder)
    create_test_files_in_folder(subfolder + "2")
    
    
    # download_s3_objects_with_prefix(BUCKET, "zipped_file/ff", "downloaded_files")

    # unzip_folder("downloaded_files/zipped_file/ffzsome-job-btest-files.zip", "/Users/zainr/deleteme/ffunzip")

    # pr_identifier = get_sanitized_pr_identifier("read-deal")
    # print(pr_identifier)
    # workflow = "test-workflow"
    # job = "test-job-name"
    # shard = "shard-3"
    # cache_dir = f"/Users/zainr/test-infra/{PYTEST_CACHE_DIR_NAME}"
    # upload_pytest_cache(pr_identifier, workflow, job, shard, cache_dir, BUCKET)

    # temp_dir = "/Users/zainr/deleteme/tmp"
    
    # cache_dir_new = f"/Users/zainr/deleteme/test_pytest_cache"
    # download_pytest_cache(pr_identifier, workflow, job, cache_dir_new, BUCKET)

