from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def get_owner_updates(
    repo: str,
    current_owners: dict[str, list[str]],
    stored_owners: dict[str, list[str]],
) -> list[tuple[str, str, list[str]]]:
    # Skip unchanged rows and use current source owners for new or changed files.
    # Clear stale owners for deleted files by writing an empty owner list.
    owner_updates = []
    for file_path in sorted(current_owners.keys() | stored_owners.keys()):
        module_owners = current_owners.get(file_path, [])
        if file_path not in stored_owners or set(module_owners) != set(
            stored_owners[file_path]
        ):
            owner_updates.append((repo, file_path, module_owners))
    return owner_updates


def main(argv: list[str] | None = None) -> None:
    endpoint = os.environ.get("CLICKHOUSE_ENDPOINT")
    username = os.environ.get("CLICKHOUSE_USERNAME")
    password = os.environ.get("CLICKHOUSE_PASSWORD")

    parser = argparse.ArgumentParser(description="Populate ClickHouse test owners")
    parser.add_argument("--repo", default="pytorch/pytorch")
    parser.add_argument(
        "--database",
        required=bool(endpoint and username and password),
        help="Target ClickHouse database",
    )
    args = parser.parse_args(argv)

    current_owners = json.loads(
        subprocess.check_output(
            [sys.executable, "scripts/codeowners/owners.py"],
            cwd=subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
            ).strip(),
            text=True,
        )
    )

    if not (endpoint and username and password):
        print(
            f"Dry run: {len(current_owners)} test files found; "
            "database not updated (missing credentials).",
            file=sys.stderr,
        )
        return

    import clickhouse_connect

    with clickhouse_connect.get_client(
        host=endpoint.removeprefix("https://").removesuffix(":8443"),
        username=username,
        password=password,
        database=args.database,
        secure=True,
        port=8443,
    ) as client:
        query_result = client.query(
            "SELECT file, owners FROM owners FINAL WHERE repo = {repo:String}",
            parameters={"repo": args.repo},
        )
        owner_updates = get_owner_updates(
            repo=args.repo,
            current_owners=current_owners,
            stored_owners=dict(query_result.result_rows),
        )
        if owner_updates:
            client.insert(
                "owners",
                owner_updates,
                column_names=["repo", "file", "owners"],
            )
        print(
            f"Found {len(current_owners)} test files; "
            f"updated {len(owner_updates)} files in {args.database}.owners",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
