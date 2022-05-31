from typing import Any
from datetime import datetime, timedelta
from gitutils import _check_output

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Print latest commits")
    parser.add_argument("--minutes", type=int, default=60, help="duration in minutes of last commits")
    return parser.parse_args()

def print_latest_commits(minutes: int = 60) -> None:
    current_time = datetime.now()
    time_since = current_time - timedelta(minutes=minutes)
    timestamp_since = datetime.timestamp(time_since)

    commits = _check_output(
        [
            "git",
            "rev-list",
            f"--max-age={timestamp_since}",
            "--all",
        ],
        encoding="ascii",
    ).splitlines()

    for commit in commits:
        print(commit)

def main() -> None:
    args = parse_args()
    print_latest_commits(args.minutes)

if __name__ == "__main__":
    main()
