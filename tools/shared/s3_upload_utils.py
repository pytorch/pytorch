import boto3
import os
import json

def zip_folder(folder_to_zip, zip_file_path):
    import shutil
    print(f"Zipping {folder_to_zip} to {zip_file_path}")
    return shutil.make_archive(zip_file_path, 'zip', folder_to_zip)


def unzip_folder(zip_file_path, unzip_to_folder):
    import shutil
    print(f"Unzipping {zip_file_path} to {unzip_to_folder}")
    return shutil.unpack_archive(zip_file_path, unzip_to_folder, 'zip')


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json_file(file_path, content):
    dir = os.path.dirname(file_path)
    ensure_dir_exists(dir)

    with open(file_path, 'w') as f:
        json.dump(content, f, indent=2)


def upload_file_to_s3(file_name, bucket, key):
    print(f"Uploading {file_name} to s3://{bucket}/{key}...", end="")

    boto3.client("s3").upload_file(
        file_name,
        bucket,
        key,
    )

    print("done")

def download_s3_objects_with_prefix(bucket, prefix, download_folder):
    s3 = boto3.resource('s3')
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