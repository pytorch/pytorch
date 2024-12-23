import argparse
from functools import cache
import os

import zipfile
import shutil
from torch.hub import Path


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
    return parser.parse_args()


@cache
def get_s3_resource():
    import boto3

    return boto3.resource("s3")


def s3_upload(s3_bucket, s3_key, file):
    s3 = get_s3_resource()
    s3.meta.client.upload_file(file, s3_bucket, s3_key)


def extract_metadata(file):
    # Copy the file to a temp location and extract the METADATA file
    file_name = Path(file).name
    tmp = "/tmp"
    shutil.copy(file, tmp)
    zip_file = f"{tmp}/{file_name.replace('.whl', '.zip')}"
    shutil.move(f"{tmp}/{file_name}", zip_file)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for filename in zip_ref.infolist():
            if filename.filename.endswith(".dist-info/METADATA"):
                filename.filename = "METADATA"
                os.remove(f"{tmp}/METADATA")
                zip_ref.extract(filename, tmp)
        return tmp


if __name__ == "__main__":
    args = parse_args()
    location = extract_metadata(args.package)
    metadata_file = f"{location}/METADATA"
    s3_upload(
        args.bucket,
        f"{args.key_prefix}/${Path(args.package).name.replace('.whl', '.metadata')}",
        metadata_file,
    )
