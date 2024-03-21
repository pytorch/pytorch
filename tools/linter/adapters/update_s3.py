"""Uploads a new binary to s3 and updates its hash in the config file.

You'll need to have appropriate credentials on the PyTorch AWS buckets, see:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
for how to configure them.
"""

import argparse
import hashlib
import json
import logging
import os

import boto3  # type: ignore[import]


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="s3 binary updater",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="path to config json that you are trying to update",
    )
    parser.add_argument(
        "--linter",
        required=True,
        help="name of linter you're trying to update",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="which platform you are uploading the binary for",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="file to upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="if set, don't actually upload/write hash",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = json.load(open(args.config_json))
    linter_config = config[args.linter][args.platform]
    bucket = linter_config["s3_bucket"]
    object_name = linter_config["object_name"]

    # Upload the file
    logging.info(
        "Uploading file %s to s3 bucket: %s, object name: %s",
        args.file,
        bucket,
        object_name,
    )
    if not args.dry_run:
        s3_client = boto3.client("s3")
        s3_client.upload_file(args.file, bucket, object_name)

    # Update hash in repo
    hash_of_new_binary = compute_file_sha256(args.file)
    logging.info("Computed new hash for binary %s", hash_of_new_binary)

    linter_config["hash"] = hash_of_new_binary
    config_dump = json.dumps(config, indent=4, sort_keys=True)

    logging.info("Writing out new config:")
    logging.info(config_dump)
    if not args.dry_run:
        with open(args.config_json, "w") as f:
            f.write(config_dump)


if __name__ == "__main__":
    main()
