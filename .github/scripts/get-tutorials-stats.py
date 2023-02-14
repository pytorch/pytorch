#!/usr/bin/env python3
import os.path
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
from pprint import pprint

access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
dynamodb = boto3.resource('dynamodb', region_name="us-east-1", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

def run_command(cmd: str, cwd: Optional[str] = None) -> str:
    """
    Run a shell command.

    Args:
        cmd: Command to run
        cwd: Working directory
    Returns:
        Output of the command.
    """
    import shlex
    from subprocess import check_output

    return check_output(shlex.split(cmd), cwd=cwd).decode("utf-8")


def get_history(cwd: Optional[str] = None) -> List[List[str]]:
    """
    Get commit history from git.
    Args:
        cwd: Working directory
    Returns:
        List of commit hashes
    """
    lines = run_command(
        "git log --date=short --pretty=format:" '%h;"%an";%ad;"%s"' " --shortstat",
        cwd=cwd,
    ).split("\n")

    def parse_string(line: str) -> str:
        """
        Parse a line into a list of strings.
        Args:
            line: Line to parse
        Returns:
            List of strings
        """
        # Add missing deletions info
        if "deletion" not in line:
            line += ", 0 deletions(-)"
        elif "insertion" not in line:
            line = ",".join(
                [line.split(",")[0], " 0 insertions(+)", line.split(",")[-1]]
            )
        return line

    def do_replace(x: str) -> str:
        """
        Replace patterns from git log with empty string. This helps us get rid of unnecessary "insertions" and "deletions"
        and we'd like to have only numbers.
        Args:   x: String to replace
        Returns:
            Replaced string
        """
        for pattern in [
            "files changed",
            "file changed",
            "insertions(+)",
            "insertion(+)",
            "deletion(-)",
            "deletions(-)",
        ]:
            x = x.replace(f" {pattern}", "")
        return x

    title = None
    rc: List[List[str]] = []
    for line in lines:
        # Check for weird entries where subject has double quotes or similar issues
        if title is None:
            title = line.split(";", 3)
        # In the lines with stat, add 0 insertions or 0 deletions to make sure we don't break the table
        elif "files changed" in line.replace("file changed", "files changed"):
            stats = do_replace(parse_string(line)).split(",")
        elif len(line) == 0:
            rc.append(title + stats)
            title = None
        else:
            rc.append(title + ["0", "0", "0"])
            title = line.split(";", 3)
    return rc


def exists(table_name: str) -> bool:
    """
    Determines whether a table exists. As a side effect, stores the table in
    a member variable.
    """
    try:
        table = dynamodb.Table(table_name)
        table.load()
        exists = True
        print("Table exists")
    except ClientError as err:
        if err.response['Error']['Code'] == 'ResourceNotFoundException':
            exists = False
        else:
            print("Unknown error")
            pprint(err.response)
    return exists

def put_data(history_log: List[List[str]], table_name: str) -> None:
    table = dynamodb.Table(table_name)
    print("Uploading data to table ...")
    for i in history_log:
        table.put_item(Item={
            'commit_id': i[0],
            'author': i[1],
            'date': i[2],
            'title': i[3],
            'number_of_changed_files': i[4],
            'lines_added': i[5],
            'lines_deleted': i[6]
        })


def main() -> None:
    tutorials_dir = os.path.expanduser("./tutorials")
    get_history_log = get_history(tutorials_dir)
    table_name = 'torchci-tutorial-metadata'
    exists(table_name)
    put_data(get_history(tutorials_dir), table_name)


if __name__ == "__main__":
    main()
