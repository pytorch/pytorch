# Helper to get the base reference of a PR if its ciflow workflow was triggered

import json
import os
import re
import sys
import time
import urllib
import urllib.parse

from typing import Any, Callable, Dict, Optional, Tuple
from urllib.request import Request, urlopen

def parse_json_and_links(conn: Any) -> Tuple[Any, Dict[str, Dict[str, str]]]:
    links = {}
    # Extract links which GH uses for pagination
    # see https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Link
    if "Link" in conn.headers:
        for elem in re.split(", *<", conn.headers["Link"]):
            try:
                url, params_ = elem.split(";", 1)
            except ValueError:
                continue
            url = urllib.parse.unquote(url.strip("<> "))
            qparams = urllib.parse.parse_qs(params_.strip(), separator=";")
            params = {
                k: v[0].strip('"')
                for k, v in qparams.items()
                if type(v) is list and len(v) > 0
            }
            params["url"] = url
            if "rel" in params:
                links[params["rel"]] = params

    return json.load(conn), links

def fetch_url(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
    retries: Optional[int] = 3,
    backoff_timeout: float = 0.5,
) -> Any:
    if headers is None:
        headers = {}
    try:
        with urlopen(Request(url, headers=headers)) as conn:
            return reader(conn)
    except urllib.error.HTTPError as err:
        if isinstance(retries, (int, float)) and retries > 0:
            time.sleep(backoff_timeout)
            return fetch_url(
                url,
                headers=headers,
                reader=reader,
                retries=retries - 1,
                backoff_timeout=backoff_timeout,
            )
        exception_message = (
            "Is github alright?",
            f"Recieved status code '{err.code}' when attempting to retrieve {url}:\n",
            f"{err.reason}\n\nheaders={err.headers}",
        )
        raise RuntimeError(exception_message) from err

def fetch_base_ref(url: str, headers: Dict[str, str]) -> Any:
    response, links = fetch_url(url, headers=headers, reader=parse_json_and_links)
    return response["base"]["ref"]


def find_base_branch() -> str:
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    GITHUB_REF_NAME = os.environ.get("GITHUB_REF_NAME", "")
    pull = 0
    if GITHUB_REF_NAME != "" and "ciflow" in GITHUB_REF_NAME:
        pull = GITHUB_REF_NAME.rsplit('/', 1)[-1]
    else:
        return "main"

    PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    REQUEST_HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + GITHUB_TOKEN,
    }

    url = f"{PYTORCH_GITHUB_API}/pulls/{pull}"
    base_ref = fetch_base_ref(url, REQUEST_HEADERS)
    if base_ref.startswith("release/"):
        return base_ref

    return "main"

def main() -> None:
    try:
        print(find_base_branch())
    except Exception as e:
        print(repr(e), file=sys.stderr)


if __name__ == "__main__":
    main()
