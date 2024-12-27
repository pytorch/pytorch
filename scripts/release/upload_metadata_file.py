import argparse
from functools import cache
import os

from pathlib import Path
import subprocess
import zipfile
import shutil


def parse_args():
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
    if args.bucket.startswith("s3://"):
        args.bucket = args.bucket[5:]
    return args


@cache
def get_s3_client():
    try:
        import boto3
    except ImportError:
        subprocess.run(["pip", "install", "boto3"])
        import boto3

    return boto3.client("s3")


def s3_upload(s3_bucket, s3_key, file, dry_run):
    s3 = get_s3_client()
    if dry_run:
        print(f"Dry run uploading {file} to s3://{s3_bucket}/{s3_key}")
        return
    s3.upload_file(file, s3_bucket, s3_key, ExtraArgs={"ChecksumAlgorithm": "sha256"})


def extract_metadata(file):
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
