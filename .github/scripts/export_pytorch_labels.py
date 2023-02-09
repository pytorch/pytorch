#!/usr/bin/env python3
'''
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

As a part of enforcing test ownership, we want to maintain a list of existing PyTorch labels
to verify the owners' existence. This script outputs a file containing a list of existing
pytorch/pytorch labels so that the file could be uploaded to S3.

This script assumes the correct env vars are set for AWS permissions.

'''

import boto3  # type: ignore[import]
import json

from label_utils import gh_get_labels


def main() -> None:
    labels_file_name = "pytorch_labels.json"
    obj = boto3.resource('s3').Object('ossci-metrics', labels_file_name)
    obj.put(Body=json.dumps(gh_get_labels()).encode())


if __name__ == '__main__':
    main()
