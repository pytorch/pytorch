
import requests
import json
import os

def comment_on_pr(text, pr_number) -> None:
    headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization" :"Bearer XXX",
            "X-GitHub-Api-Version": "2022-11-28"
            }

    response = requests.post(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{pr_number}/comments",
        data=json.dumps({"body": text}),
        headers=headers,
    )
    print(response)


if __name__ == "__main__":

    comment_on_pr("this comment is auto generated testing , ",131751 )
