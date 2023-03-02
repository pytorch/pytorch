#!/usr/bin/env python3
# Checks if PR updates submodules by querying GitHub GraphQL

from typing import Any, List, Tuple
from trymerge import gh_graphql, GH_GET_PR_NEXT_FILES_QUERY

GH_GET_PR_FILES_AND_SUBMODULES = """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    submodules(first: 100) {
      nodes {
        path
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
    pullRequest(number: $number) {
      title
      body
      labels(first: 100) {
        nodes {
          name
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      files(first: 100) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"""

def gh_get_pr_files_and_submodules(owner: str, name: str, number: int) -> Tuple[List[str], List[str]]:
    rc = gh_graphql(GH_GET_PR_FILES_AND_SUBMODULES, name=name, owner=owner, number=number)["data"]["repository"]
    submodules = [x["path"] for x in rc["submodules"]["nodes"]]
    files = [x["path"] for x in rc["pullRequest"]["files"]["nodes"]]
    return (files, submodules)

def gh_get_pr_updated_submodules(owner: str, name: str, number: int) -> List[str]:
    (files, submodules) = gh_get_pr_files_and_submodules(owner=owner, name=name, number=number)
    return [f for f in files if f in submodules]

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Check if PR updates submodules")
    parser.add_argument("--owner", type=str, default="pytorch")
    parser.add_argument("--name", type=str, default="pytorch")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(gh_get_pr_updated_submodules(name=args.name, owner=args.owner, number=args.pr_num))


if __name__ == "__main__":
    main()
