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
from typing import Any

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload metadata file to S3")
    parser.add_argument(
        "--package", type=str, required=True, help="Path to the package"
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
    s3.upload_file(
        file,
        s3_bucket,
        s3_key,
        ExtraArgs={"ChecksumAlgorithm": "sha256", "ACL": "public-read"},
    )


def extract_metadata(file: str) -> str:
    # Copy the file to a temp location to extract the METADATA file
    file_name = Path(file).name
    tmp = "/tmp"
    shutil.copy(file, tmp)
    zip_file = f"{tmp}/{file_name.replace('.whl', '.zip')}"
    shutil.move(f"{tmp}/{file_name}", zip_file)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for filename in zip_ref.infolist():
            if filename.filename.endswith(".dist-info/METADATA"):
                filename.filename = "METADATA"
                if os.path.exists(f"{tmp}/METADATA"):
                    os.remove(f"{tmp}/METADATA")
                zip_ref.extract(filename, tmp)
        return tmp


if __name__ == "__main__":
    # https://peps.python.org/pep-0658/
    # Upload the METADATA file to S3
    args = parse_args()
    location = extract_metadata(args.package)
    metadata_file = f"{location}/METADATA"
    s3_upload(
        args.bucket,
        f"{args.key_prefix}/{Path(args.package).name}.metadata",
        metadata_file,
        args.dry_run,
    )
