#!/usr/bin/env -S uv run —verbose
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "boto3",
# ]
# ///
import argparse
import os
import shutil
import zipfile
from functools import cache
from pathlib import Path
from typing import Any, Optional

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload metadata file to S3")
    package = parser.add_mutually_exclusive_group(required=True)
    package.add_argument("--package", type=str, help="Path to the local package")
    package.add_argument(
        "--use-s3-prefix",
        action="store_true",
        help="Upload metadata for all wheels in the prefix specified by --key-prefix",
    )
    parser.add_argument(
        "--bucket", type=str, required=True, help="S3 bucket to upload metadata file to"
    )
    parser.add_argument(
        "--key-prefix",
        type=str,
        required=True,
        help="S3 key to upload metadata file to",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()
    # Sanitize the input a bit by removing s3:// prefix + trailing/leading
    # slashes
    if args.bucket.startswith("s3://"):
        args.bucket = args.bucket[5:]
    args.bucket = args.bucket.strip("/")
    args.key_prefix = args.key_prefix.strip("/")
    return args


@cache
def get_s3_client() -> Any:
    return boto3.client("s3")


def s3_upload(s3_bucket: str, s3_key: str, file: str, dry_run: bool) -> None:
    s3 = get_s3_client()
    if dry_run:
        print(f"Dry run uploading {file} to s3://{s3_bucket}/{s3_key}")
        return
    print(f"Uploading {file} to s3://{s3_bucket}/{s3_key}")
    s3.upload_file(
        file,
        s3_bucket,
        s3_key,
        ExtraArgs={"ChecksumAlgorithm": "sha256", "ACL": "public-read"},
    )


def copy_to_tmp(file: str) -> str:
    # Copy file with path a/b/c.d to /tmp/c.d
    file_name = Path(file).name
    tmp = "/tmp"
    shutil.copy(file, tmp)
    return f"{tmp}/{file_name}"


def extract_metadata(file: str) -> Optional[str]:
    # Extract the METADATA file from the wheel. With input file a/b/c.whl, tmp
    # is expected to have /tmp/c.whl, which gets converted to /tmp/c.zip, and
    # the METADATA file is extracted to /tmp/METADATA
    file_name = Path(file).name
    tmp = "/tmp"
    zip_file = f"{tmp}/{file_name.replace('.whl', '.zip')}"
    shutil.move(f"{tmp}/{file_name}", zip_file)

    if os.path.exists(f"{tmp}/METADATA"):
        os.remove(f"{tmp}/METADATA")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for filename in zip_ref.infolist():
            if filename.filename.endswith(".dist-info/METADATA"):
                filename.filename = "METADATA"
                zip_ref.extract(filename, tmp)
                return f"{tmp}/METADATA"
    return None


def download_package_from_s3(bucket: str, key_prefix: str, file: str) -> str:
    # Download the package from S3 to /tmp.  With input bucket a, key_prefix b,
    # and file c.d, the file located at s3://a/b/c.d is downloaded to /tmp/c.d
    s3 = get_s3_client()
    local_file = f"/tmp/{file}"
    s3.download_file(bucket, f"{key_prefix}/{file}", local_file)
    return local_file


def upload_all_metadata_in_prefix(bucket: str, prefix: str, dry_run: bool) -> None:
    # For all wheels in the prefix, upload the metadata file to S3.  This is
    # used for backfilling
    all_files = [
        file["Key"]
        for file in get_s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix)[
            "Contents"
        ]
    ]
    whls = [file for file in all_files if file.endswith(".whl")]
    metadatas = [file for file in all_files if file.endswith(".metadata")]
    for whl in whls:
        if not f"{whl}.metadata" in metadatas:
            _prefix = "/".join(whl.split("/")[:-1])
            _key = whl.split("/")[-1]
            local_file = download_package_from_s3(bucket, _prefix, _key)
            metadata_file = extract_metadata(local_file)
            if not metadata_file:
                print(f"Failed to extract metadata from {whl}")
                continue
            s3_upload(bucket, f"{whl}.metadata", metadata_file, dry_run)


if __name__ == "__main__":
    # https://peps.python.org/pep-0658/
    # Upload the METADATA file to S3
    # This script is used to upload the METADATA file to S3
    # If --package is used, it is assumed to be local and will copy the package
    # to /tmp, convert to a zip, extract the METADATA file and upload it to S3.
    # If --use-s3-prefix is used, it will backfill for all wheels in this
    # prefix.
    args = parse_args()
    if args.use_s3_prefix:
        upload_all_metadata_in_prefix(args.bucket, args.key_prefix, args.dry_run)
        exit(0)
    copy_to_tmp(args.package)
    metadata_file = extract_metadata(args.package)
    if not metadata_file:
        print(f"Failed to extract metadata from {args.package}")
        exit(0)
    s3_upload(
        args.bucket,
        f"{args.key_prefix}/{Path(args.package).name}.metadata",
        metadata_file,
        args.dry_run,
    )
