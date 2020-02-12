#!/usr/bin/env python

import os
import sys
import boto3

IMAGE_COMMIT_TAG = os.getenv('IMAGE_COMMIT_TAG')

session = boto3.session.Session()
s3 = session.resource('s3')
with open(sys.argv[1], 'rb') as data:
    s3.Bucket('ossci-windows-build').put_object(Key='pytorch/' + IMAGE_COMMIT_TAG + '.7z', Body=data)
object_acl = s3.ObjectAcl('ossci-windows-build', 'pytorch/' + IMAGE_COMMIT_TAG + '.7z')
response = object_acl.put(ACL='public-read')
