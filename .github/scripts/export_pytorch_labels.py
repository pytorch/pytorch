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
from functools import lru_cache
from typing import List, Any
from urllib.request import urlopen, Request

# Modified from https://github.com/pytorch/pytorch/blob/b00206d4737d1f1e7a442c9f8a1cadccd272a386/torch/hub.py#L129
def _read_url(url: Any) -> Any:
    with urlopen(url) as r:
        return r.headers, r.read().decode(r.headers.get_content_charset('utf-8'))


def request_for_labels(url: str) -> Any:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    return _read_url(Request(url, headers=headers))


def get_last_page(header: Any) -> int:
    # Link info looks like: <https://api.github.com/repositories/65600975/labels?per_page=100&page=2>;
    # rel="next", <https://api.github.com/repositories/65600975/labels?per_page=100&page=3>; rel="last"
    link_info = header['link']
    prefix = "&page="
    suffix = ">;"
    return int(link_info[link_info.rindex(prefix) + len(prefix):link_info.rindex(suffix)])


def update_labels(labels: List[str], info: str) -> None:
    labels_json = json.loads(info)
    labels.extend([x["name"] for x in labels_json])


@lru_cache()
def get_pytorch_labels() -> List[str]:
    prefix = "https://api.github.com/repos/pytorch/pytorch/labels?per_page=100"
    header, info = request_for_labels(prefix + "&page=1")
    labels: List[str] = []
    update_labels(labels, info)

    last_page = get_last_page(header)
    assert last_page > 0, "Error reading header info to determine total number of pages of labels"
    for page_number in range(2, last_page + 1):  # skip page 1
        _, info = request_for_labels(prefix + f"&page={page_number}")
        update_labels(labels, info)

    return labels


def send_labels_to_S3(labels: List[str]) -> None:
    labels_file_name = "pytorch_labels.json"
    obj = boto3.resource('s3').Object('ossci-metrics', labels_file_name)
    obj.put(Body=json.dumps(labels).encode())


def main() -> None:
    send_labels_to_S3(get_pytorch_labels())


if __name__ == '__main__':
    main()
