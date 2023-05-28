import hashlib
import os
from pathlib import Path
from typing import Dict, NamedTuple

from file_io_utils import (
    copy_file,
    download_s3_objects_with_prefix,
    load_json_file,
    sanitize_for_s3,
    unzip_folder,
    upload_file_to_s3,
    write_json_file,
    zip_folder,
)

PYTEST_CACHE_KEY_PREFIX = "pytest_cache"
PYTEST_CACHE_DIR_NAME = ".pytest_cache"
BUCKET = "gha-artifacts"
LASTFAILED_FILE_PATH = Path("v/cache/lastfailed")

# Temp folders
ZIP_UPLOAD = "zip-upload"
CACHE_ZIP_DOWNLOADS = "cache-zip-downloads"
UNZIPPED_CACHES = "unzipped-caches"


# Since the pr identifier can be based on include user defined text (like a branch name)
# we hash it to sanitize the input and avoid corner cases
class PRIdentifier(str):
    def __new__(cls, value: str) -> "PRIdentifier":
        md5 = hashlib.md5(value.encode("utf-8")).hexdigest()
        return super().__new__(cls, md5)


class GithubRepo(NamedTuple):
    owner: str
    name: str

    # Create a Repo from a string like "owner/repo"
    @classmethod
    def from_string(cls, repo_string: str) -> "GithubRepo":
        if "/" not in repo_string:
            raise ValueError(
                f"repo_string must be of the form 'owner/repo', not {repo_string}"
            )

        owner, name = repo_string.split("/")
        return cls(owner, name)

    def __str__(self) -> str:
        return f"{self.owner}/{self.name}"


def upload_pytest_cache(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    shard: str,
    cache_dir: Path,
    temp_dir: Path,
    bucket: str = BUCKET,
) -> None:
    """
    Uploads the pytest cache to S3, merging it with any previous caches from previous runs of the same job.
    In particular, this keeps all the failed tests across all runs of this job in the cache, so that
    future jobs that download this cache will prioritize running tests that have failed in the past.

    Args:
        pr_identifier: A unique, human readable identifier for the PR
        job: The name of the job that is uploading the cache
    """

    if not isinstance(pr_identifier, PRIdentifier):
        raise ValueError(
            f"pr_identifier must be of type PRIdentifier, not {type(pr_identifier)}"
        )

    if not bucket:
        bucket = BUCKET

    # Merge the current cache with any caches from previous runs before uploading
    # We only need to merge it with the cache for the same shard (which will have already been downloaded if it exists)
    # since the other shards will handle themselves
    shard_cache_path = _get_temp_cache_dir_path(
        temp_dir, pr_identifier, repo, job_identifier, shard
    )

    if shard_cache_path.is_dir():
        _merge_pytest_caches(shard_cache_path, cache_dir)

    #
    # Upload the cache
    #

    obj_key_prefix = _get_s3_key_prefix(pr_identifier, repo, job_identifier, shard)
    # This doesn't include the zip file extension. That'll get added later
    zip_file_path = temp_dir / ZIP_UPLOAD / obj_key_prefix

    zip_file_path = zip_folder(cache_dir, zip_file_path)
    obj_key = f"{obj_key_prefix}{os.path.splitext(zip_file_path)[1]}"  # Keep the new file extension
    upload_file_to_s3(zip_file_path, bucket, obj_key)


def download_pytest_cache(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    dest_cache_dir: Path,
    temp_dir: Path,
    bucket: str = BUCKET,
) -> None:
    """
    Downloads the pytest cache from S3. The goal is to detect any tests that have failed in the past
    and run them first, so that the dev can get faster feedback on them.

    We merge the cache from all shards since tests can get shuffled around from one shard to another
    (based on when we last updated our stats on how long each test takes to run). This ensures that
    even if a test moves to a different shard, that shard will know to run it first if had failed previously.
    """
    if not bucket:
        bucket = BUCKET

    if not isinstance(pr_identifier, PRIdentifier):
        raise ValueError(
            f"pr_identifier must be of type PRIdentifier, not {type(pr_identifier)}"
        )

    obj_key_prefix = _get_s3_key_prefix(pr_identifier, repo, job_identifier)

    zip_download_dir = temp_dir / CACHE_ZIP_DOWNLOADS / obj_key_prefix

    # downloads the cache zips for all shards
    downloads = download_s3_objects_with_prefix(
        bucket, obj_key_prefix, zip_download_dir
    )

    for downloaded_zip in downloads:
        # the file name of the zip is the shard id
        shard = os.path.splitext(os.path.basename(downloaded_zip))[0]
        cache_dir_for_shard = _get_temp_cache_dir_path(
            temp_dir, pr_identifier, repo, job_identifier, shard
        )

        unzip_folder(downloaded_zip, cache_dir_for_shard)
        print(
            f"Merging cache for job_identifier `{job_identifier}`, shard `{shard}` into `{dest_cache_dir}`"
        )
        _merge_pytest_caches(cache_dir_for_shard, dest_cache_dir)


def _get_temp_cache_dir_path(
    temp_dir: Path,
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    shard: str,
) -> Path:
    return (
        temp_dir
        / UNZIPPED_CACHES
        / _get_s3_key_prefix(pr_identifier, repo, job_identifier, shard)
        / PYTEST_CACHE_DIR_NAME
    )


def _get_s3_key_prefix(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    shard: str = "",
) -> str:
    """
    The prefix to any S3 object key for a pytest cache. It's only a prefix though, not a full path to an object.
    For example, it won't include the file extension.
    """
    prefix = f"{PYTEST_CACHE_KEY_PREFIX}/{repo.owner}/{repo.name}/{pr_identifier}/{sanitize_for_s3(job_identifier)}"

    if shard:
        prefix += f"/{shard}"

    return prefix


def _merge_pytest_caches(
    pytest_cache_dir_to_merge_from: Path, pytest_cache_dir_to_merge_into: Path
) -> None:
    # LASTFAILED_FILE_PATH is the only file we actually care about in the cache
    # since it contains all the tests that failed.
    #
    # The remaining files are static supporting files that don't really matter. They
    # make the cache folder play nice with other tools devs tend to use (e.g. git).
    # But since pytest doesn't recreate these files if the .pytest_cache folder already exists,
    # we'll copy them over as a way to protect against future bugs where a certain tool
    # may need those files to exist to work properly (their combined file size is negligible)
    static_files_to_copy = [
        ".gitignore",
        "CACHEDIR.TAG",
        "README.md",
    ]

    # Copy over the static files. These files never change, so only copy them
    # if they don't already exist in the new cache
    for static_file in static_files_to_copy:
        source_file = pytest_cache_dir_to_merge_from / static_file
        if not source_file.is_file():
            continue

        dest_file = pytest_cache_dir_to_merge_into / static_file
        if not dest_file.exists():
            copy_file(source_file, dest_file)

    # Handle the v/cache/lastfailed file
    _merge_lastfailed_files(
        pytest_cache_dir_to_merge_from, pytest_cache_dir_to_merge_into
    )


def _merge_lastfailed_files(source_pytest_cache: Path, dest_pytest_cache: Path) -> None:
    # Simple cases where one of the files doesn't exist
    source_lastfailed_file = source_pytest_cache / LASTFAILED_FILE_PATH
    dest_lastfailed_file = dest_pytest_cache / LASTFAILED_FILE_PATH

    if not source_lastfailed_file.exists():
        return
    if not dest_lastfailed_file.exists():
        copy_file(source_lastfailed_file, dest_lastfailed_file)
        return

    # Both files exist, so we need to merge them
    from_lastfailed = load_json_file(source_lastfailed_file)
    to_lastfailed = load_json_file(dest_lastfailed_file)
    merged_content = _merged_lastfailed_content(from_lastfailed, to_lastfailed)

    # Save the results
    write_json_file(dest_lastfailed_file, merged_content)


def _merged_lastfailed_content(
    from_lastfailed: Dict[str, bool], to_lastfailed: Dict[str, bool]
) -> Dict[str, bool]:
    """
    The lastfailed files are dictionaries where the key is the test identifier.
    Each entry's value appears to always be `true`, but let's not count on that.
    An empty dictionary is represented with a single value with an empty string as the key.
    """

    # If an entry in from_lastfailed doesn't exist in to_lastfailed, add it and it's value
    for key in from_lastfailed:
        if key not in to_lastfailed:
            to_lastfailed[key] = from_lastfailed[key]

    if len(to_lastfailed) > 1:
        # Remove the empty entry if it exists since we have actual entries now
        if "" in to_lastfailed:
            del to_lastfailed[""]

    return to_lastfailed
