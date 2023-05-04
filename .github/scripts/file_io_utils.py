import json
import os
import re
import shutil

import boto3


def zip_folder(folder_to_zip, dest_file_base_name):
    """
    Returns the path to the resulting zip file, with the appropriate extension added if needed
    """
    # shutil.make_archive will append .zip to the dest_file_name, so we need to remove it if it's already there
    if dest_file_base_name.endswith(".zip"):
        dest_file_base_name = dest_file_base_name[:-4]

    ensure_dir_exists(os.path.dirname(dest_file_base_name))

    print(f"Zipping {folder_to_zip} to {dest_file_base_name}")
    return shutil.make_archive(dest_file_base_name, "zip", folder_to_zip)


def unzip_folder(zip_file_path, unzip_to_folder):
    """
    Returns the path to the unzipped folder
    """
    print(f"Unzipping {zip_file_path} to {unzip_to_folder}")
    shutil.unpack_archive(zip_file_path, unzip_to_folder, "zip")


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_json_file(file_path):
    """
    Returns the deserialized json object
    """
    with open(file_path, "r") as f:
        return json.load(f)


def write_json_file(file_path, content):
    dir = os.path.dirname(file_path)
    ensure_dir_exists(dir)

    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)


def sanitize_for_s3(text):
    """
    S3 keys can only contain alphanumeric characters, underscores, and dashes.
    This function replaces all other characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def upload_file_to_s3(file_name, bucket, key):
    print(f"Uploading {file_name} to s3://{bucket}/{key}...", end="")

    boto3.client("s3").upload_file(
        file_name,
        bucket,
        key,
    )

    print("done")


def download_s3_objects_with_prefix(bucket, prefix, download_folder):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)

    downloads = []

    for obj in bucket.objects.filter(Prefix=prefix):
        download_path = os.path.join(download_folder, obj.key)
        ensure_dir_exists(os.path.dirname(download_path))
        print(f"Downloading s3://{bucket.name}/{obj.key} to {download_path}...", end="")

        s3.Object(bucket.name, obj.key).download_file(download_path)
        downloads.append(download_path)
        print("done")

    return downloads
