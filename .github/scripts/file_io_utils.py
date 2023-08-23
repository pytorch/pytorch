import json
import re
import shutil
from pathlib import Path
from typing import Any, List

import boto3  # type: ignore[import]


def zip_folder(folder_to_zip: Path, dest_file_base_name: Path) -> Path:
    """
    Returns the path to the resulting zip file, with the appropriate extension added if needed
    """
    # shutil.make_archive will append .zip to the dest_file_name, so we need to remove it if it's already there
    if dest_file_base_name.suffix == ".zip":
        dest_file_base_name = dest_file_base_name.with_suffix("")

    ensure_dir_exists(dest_file_base_name.parent)

    print(f"Zipping {folder_to_zip}\n     to {dest_file_base_name}")
    # Convert to string because shutil.make_archive doesn't like Path objects
    return Path(shutil.make_archive(str(dest_file_base_name), "zip", folder_to_zip))


def unzip_folder(zip_file_path: Path, unzip_to_folder: Path) -> None:
    """
    Returns the path to the unzipped folder
    """
    print(f"Unzipping {zip_file_path}")
    print(f"       to {unzip_to_folder}")
    shutil.unpack_archive(zip_file_path, unzip_to_folder, "zip")


def ensure_dir_exists(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def copy_file(source_file: Path, dest_file: Path) -> None:
    ensure_dir_exists(dest_file.parent)
    shutil.copyfile(source_file, dest_file)


def load_json_file(file_path: Path) -> Any:
    """
    Returns the deserialized json object
    """
    with open(file_path, "r") as f:
        return json.load(f)


def write_json_file(file_path: Path, content: Any) -> None:
    dir = file_path.parent
    ensure_dir_exists(dir)

    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)


def sanitize_for_s3(text: str) -> str:
    """
    S3 keys can only contain alphanumeric characters, underscores, and dashes.
    This function replaces all other characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def upload_file_to_s3(file_name: Path, bucket: str, key: str) -> None:
    print(f"Uploading {file_name}")
    print(f"       to s3://{bucket}/{key}")

    boto3.client("s3").upload_file(
        str(file_name),
        bucket,
        key,
    )


def download_s3_objects_with_prefix(
    bucket_name: str, prefix: str, download_folder: Path
) -> List[Path]:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    downloads = []

    for obj in bucket.objects.filter(Prefix=prefix):
        download_path = download_folder / obj.key

        ensure_dir_exists(download_path.parent)
        print(f"Downloading s3://{bucket.name}/{obj.key}")
        print(f"         to {download_path}")

        s3.Object(bucket.name, obj.key).download_file(str(download_path))
        downloads.append(download_path)

    if len(downloads) == 0:
        print(
            f"There were no files matching the prefix `{prefix}` in bucket `{bucket.name}`"
        )

    return downloads
