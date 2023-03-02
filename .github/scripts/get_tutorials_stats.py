#!/usr/bin/env python3
import os.path
from typing import List, Optional, Tuple, Dict, Union
import boto3
from botocore.exceptions import ClientError
from pprint import pprint

dynamodb = boto3.resource('dynamodb', region_name="us-east-1")

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

def get_file_names(cwd: Optional[str] = None) -> List[Tuple[str, List[Tuple[str, int, int]]]]:
    lines = run_command('git log --pretty="format:%h" --numstat', cwd=cwd).split("\n")
    rc = []
    commit_hash = ""
    files: List[Tuple[str, int, int]] = []
    for line in lines:
        if not line:
            # Git log uses empty line as separator between commits (except for oneline case)
            rc.append((commit_hash, files))
            commit_hash, files = "", []
        elif not commit_hash:
            # First line is commit short hash
            commit_hash = line
        elif len(line.split("\t")) != 3:
            # Encountered an empty commit
            assert(len(files) == 0)
            rc.append((commit_hash, files))
            commit_hash = line
        else:
            added, deleted, name = line.split("\t")
            # Special casing for binary files
            if added == "-":
                assert deleted == "-"
                files.append((name, -1, -1))
            else:
                files.append((name, int(added), int(deleted)))
    return rc

def table_exists(table_name: str) -> bool:
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

def delete_table(table_name: str) -> None:
    table = dynamodb.Table(table_name)
    table.delete()

    print(f"Deleting {table.name}...")
    table.wait_until_not_exists()

def create_table(table_name: str) -> None:
    """
    Creates a DynamoDB table.

    :param dyn_resource: Either a Boto3 or DAX resource.
    :return: The newly created table.
    """

    table_name = table_name
    params = {
        'TableName': table_name,
        'KeySchema': [
            {'AttributeName': 'commit_id', 'KeyType': 'HASH'},
            {'AttributeName': 'date', 'KeyType': 'RANGE'}
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'commit_id', 'AttributeType': 'S'},
            {'AttributeName': 'date', 'AttributeType': 'S'}
        ],
        'ProvisionedThroughput': {
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    }
    table = dynamodb.create_table(**params)
    print(f"Creating {table_name}...")
    table.wait_until_exists()

def create_table2(table_name: str) -> None:
    """
    Creates a DynamoDB table.

    :param dyn_resource: Either a Boto3 or DAX resource.
    :return: The newly created table.
    """

    table_name = table_name
    params = {
        'TableName': table_name,
        'KeySchema': [
            {'AttributeName': 'commit_id', 'KeyType': 'HASH'},
            {'AttributeName': 'filename', 'KeyType': 'RANGE'}
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'commit_id', 'AttributeType': 'S'},
            {'AttributeName': 'filename', 'AttributeType': 'S'}
        ],
        'ProvisionedThroughput': {
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    }
    table = dynamodb.create_table(**params)
    print(f"Creating {table_name}...")
    table.wait_until_exists()

def convert_to_dict(entry: Tuple[str, List[Tuple[str,int, int]]]) -> List[Dict[str, Union[str, int]]]:
    return [
        { 'commit_id': entry[0], 'filename': i[0], 'lines_added': i[1], 'lines_deleted': i[2]}
        for i in entry[1]
    ]

def main() -> None:
    tutorials_dir = os.path.expanduser("./tutorials")
    get_history_log = get_history(tutorials_dir)
    commits_to_files = get_file_names(tutorials_dir)
    table_name_history = 'torchci-tutorial-metadata'
    table_name_filenames = "torchci-tutorial-filenames"
    table_history = dynamodb.Table(table_name_history)
    table_filenames = dynamodb.Table(table_name_filenames)
    table_exists(table_name_history)
    table_exists(table_name_filenames)
    print("Uploading data to {table_name_history}")
    for i in get_history_log:
        table_history.put_item(Item={
            'commit_id': i[0],
            'author': i[1],
            'date': i[2],
            'title': i[3],
            'number_of_changed_files': int(i[4]),
            'lines_added': int(i[5]),
            'lines_deleted': int(i[6])
        })
    print("Finished uploading data to {table_name_history}")
    print("Uploading data to {table_name_filenames}")
    for entry in commits_to_files:
        items = convert_to_dict(entry)
        for item in items:
            table_filenames.put_item(Item=item)
    print("Finished uploading data to {table_name_filenames}")
    print("Success!")

if __name__ == "__main__":
    main()
