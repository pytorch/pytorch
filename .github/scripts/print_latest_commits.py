import sys
import subprocess
from datetime import datetime
from datetime import timedelta

def print_latest_commits(mins = 60):
    current_time = datetime.now()
    time_since = current_time - timedelta(minutes=mins)
    timestamp_since = datetime.timestamp(time_since)

    commits = subprocess.check_output(
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
    print_latest_commits() if len(sys.argv) == 1 else print_latest_commits(int(sys.argv[1]))

if __name__ == "__main__":
    main()
