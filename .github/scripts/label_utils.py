"""GitHub Label Utilities."""

import json

from functools import lru_cache
from typing import List, Any, Tuple
from urllib.request import urlopen, Request

# Modified from https://github.com/pytorch/pytorch/blob/b00206d4737d1f1e7a442c9f8a1cadccd272a386/torch/hub.py#L129
def _read_url(url: Request) -> Tuple[Any, Any]:
    with urlopen(url) as r:
        return r.headers, r.read().decode(r.headers.get_content_charset('utf-8'))


def request_for_labels(url: str) -> Tuple[Any, Any]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    return _read_url(Request(url, headers=headers))


def update_labels(labels: List[str], info: str) -> None:
    labels_json = json.loads(info)
    labels.extend([x["name"] for x in labels_json])


def get_last_page_num_from_header(header: Any) -> int:
    # Link info looks like: <https://api.github.com/repositories/65600975/labels?per_page=100&page=2>;
    # rel="next", <https://api.github.com/repositories/65600975/labels?per_page=100&page=3>; rel="last"
    link_info = header['link']
    prefix = "&page="
    suffix = ">;"
    return int(link_info[link_info.rindex(prefix) + len(prefix):link_info.rindex(suffix)])


@lru_cache()
def gh_get_labels(org: str, repo: str) -> List[str]:
    prefix = f"https://api.github.com/repos/{org}/{repo}/labels?per_page=100"
    header, info = request_for_labels(prefix + "&page=1")
    labels: List[str] = []
    update_labels(labels, info)

    last_page = get_last_page_num_from_header(header)
    assert last_page > 0, "Error reading header info to determine total number of pages of labels"
    for page_number in range(2, last_page + 1):  # skip page 1
        _, info = request_for_labels(prefix + f"&page={page_number}")
        update_labels(labels, info)

    return labels
