from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import urllib.parse
from typing import Any, Callable, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.stats.upload_stats_lib import upload_to_s3


FILTER_OUT_USERS = {
    "pytorchmergebot",
    "facebook-github-bot",
    "pytorch-bot[bot]",
    "pytorchbot",
    "pytorchupdatebot",
    "dependabot[bot]",
}


def _fetch_url(
    url: str,
    headers: dict[str, str],
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Any:
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return reader(conn)
    except HTTPError as err:
        print(err.reason)
        print(err.headers)
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"Rate limit exceeded: {err.headers['X-RateLimit-Used']}/{err.headers['X-RateLimit-Limit']}"
            )
        raise


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={urllib.parse.quote(str(val))}" for name, val in params.items()
        )
    return cast(
        list[dict[str, Any]],
        _fetch_url(url, headers=headers, data=data, reader=json.load),
    )


def get_external_pr_data(
    start_date: datetime.date, end_date: datetime.date, period_length: int = 1
) -> list[dict[str, Any]]:
    pr_info = []
    period_begin_date = start_date

    pr_count = 0
    users: set[str] = set()
    while period_begin_date < end_date:
        period_end_date = period_begin_date + datetime.timedelta(days=period_length - 1)
        page = 1
        responses: list[dict[str, Any]] = []
        while len(responses) > 0 or page == 1:
            response = cast(
                dict[str, Any],
                fetch_json(
                    "https://api.github.com/search/issues",  # @lint-ignore
                    params={
                        "q": f'repo:pytorch/pytorch is:pr is:closed \
                            label:"open source" label:Merged -label:Reverted closed:{period_begin_date}..{period_end_date}',
                        "per_page": "100",
                        "page": str(page),
                    },
                ),
            )
            items = response["items"]
            for item in items:
                u = item["user"]["login"]
                if u not in FILTER_OUT_USERS:
                    pr_count += 1
                    users.add(u)
            page += 1

        pr_info.append(
            {
                "date": str(period_begin_date),
                "pr_count": pr_count,
                "user_count": len(users),
                "users": list(users),
            }
        )
        period_begin_date = period_end_date + datetime.timedelta(days=1)
    return pr_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload external contribution stats to s3"
    )
    parser.add_argument(
        "--startDate",
        type=datetime.date.fromisoformat,
        required=True,
        help="the first date to upload data for in any valid ISO 8601 format format (eg. YYYY-MM-DD).",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=False,
        help="the number of days to upload data for. Default is 1.",
        default=1,
    )
    parser.add_argument(
        "--period-length",
        type=int,
        required=False,
        help="the number of days to group data for. Default is 1.",
        default=1,
    )
    args = parser.parse_args()
    for i in range(args.length):
        tries = 0
        startdate = args.startDate + datetime.timedelta(days=i)
        data = get_external_pr_data(
            startdate,
            startdate + datetime.timedelta(days=args.period_length),
            period_length=args.period_length,
        )
        for pr_info in data:
            # sometimes users does not get added, so we check it got uploaded
            assert "users" in pr_info
            assert isinstance(pr_info["users"], list)
        print(f"uploading the following data: \n {data}")
        upload_to_s3(
            bucket_name="torchci-contribution-data",
            key=f"external_contribution_counts/{str(startdate)}",
            docs=data,
        )
        # get around rate limiting
        time.sleep(10)
