#!/usr/bin/env python3

import os
import re


def set_output(name: str, val: str) -> None:
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    ref = os.environ["GITHUB_REF"]
    m = re.match(r"^refs/(\w+)/(.*)$", ref)
    if m:
        category, stripped = m.groups()
        if category == "heads":
            set_output("branch", stripped)
        elif category == "pull":
            set_output("branch", "pull/" + stripped.split("/")[0])
        elif category == "tags":
            set_output("tag", stripped)


if __name__ == "__main__":
    main()
