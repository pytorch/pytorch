import argparse
import datetime

import time
from typing import Any, Dict, List, Set

from tools.github.github_utils import gh_fetch_json_dict

from tools.stats.upload_stats_lib import upload_to_s3

FILTER_OUT_USERS = {
    "pytorchmergebot",
    "facebook-github-bot",
    "pytorch-bot[bot]",
    "pytorchbot",
    "pytorchupdatebot",
    "dependabot[bot]",
}


def get_external_pr_data(
    start_date: datetime.date, end_date: datetime.date, period_length: int = 1
) -> List[Dict[str, Any]]:
    pr_info = []
    period_begin_date = start_date

    pr_count = 0
    users: Set[str] = set()
    while period_begin_date < end_date:
        period_end_date = period_begin_date + datetime.timedelta(days=period_length - 1)
        page = 1
        responses: List[Dict[str, Any]] = []
        while len(responses) > 0 or page == 1:
            response = gh_fetch_json_dict(
                url="https://api.github.com/search/issues",
                params={
                    "q": f'repo:pytorch/pytorch is:pr is:closed \
                            label:"open source" label:Merged -label:Reverted closed:{period_begin_date}..{period_end_date}',
                    "per_page": "100",
                    "page": str(page),
                },
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
        description="Upload external contribution stats to Rockset"
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
