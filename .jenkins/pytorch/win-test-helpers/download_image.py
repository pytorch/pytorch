#!/usr/bin/env python

import os
import sys
import boto3
import botocore

IMAGE_COMMIT_TAG = os.getenv('IMAGE_COMMIT_TAG')

session = boto3.session.Session()
s3 = session.resource('s3')
BUCKET_NAME = 'ossci-windows-build'
KEY = 'pytorch/' + IMAGE_COMMIT_TAG + '.7z'
LOCAL_FILE_PATH = sys.argv[1]
try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, LOCAL_FILE_PATH)
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
