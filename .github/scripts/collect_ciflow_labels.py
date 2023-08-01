#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Any, cast, Dict, List, Set

import yaml

GITHUB_DIR = Path(__file__).parent.parent


def get_workflows_push_tags() -> Set[str]:
    "Extract all known push tags from workflows"
    rc: Set[str] = set()
    for fname in (GITHUB_DIR / "workflows").glob("*.yml"):
        with fname.open("r") as f:
            wf_yml = yaml.safe_load(f)
        # "on" is alias to True in yaml
        on_tag = wf_yml.get(True, None)
        push_tag = on_tag.get("push", None) if isinstance(on_tag, dict) else None
        tags_tag = push_tag.get("tags", None) if isinstance(push_tag, dict) else None
        if isinstance(tags_tag, list):
            rc.update(tags_tag)
    return rc


def filter_ciflow_tags(tags: Set[str]) -> List[str]:
    "Return sorted list of ciflow tags"
    return sorted(
        tag[:-2] for tag in tags if tag.startswith("ciflow/") and tag.endswith("/*")
    )


def read_probot_config() -> Dict[str, Any]:
    with (GITHUB_DIR / "pytorch-probot.yml").open("r") as f:
        return cast(Dict[str, Any], yaml.safe_load(f))


def update_probot_config(labels: Set[str]) -> None:
    orig = read_probot_config()
    orig["ciflow_push_tags"] = filter_ciflow_tags(labels)
    with (GITHUB_DIR / "pytorch-probot.yml").open("w") as f:
        yaml.dump(orig, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Validate or update list of tags")
    parser.add_argument("--validate-tags", action="store_true")
    args = parser.parse_args()
    pushtags = get_workflows_push_tags()
    if args.validate_tags:
        config = read_probot_config()
        ciflow_tags = set(filter_ciflow_tags(pushtags))
        config_tags = set(config["ciflow_push_tags"])
        if config_tags != ciflow_tags:
            print("Tags mismatch!")
            if ciflow_tags.difference(config_tags):
                print(
                    "Reference in workflows but not in config",
                    ciflow_tags.difference(config_tags),
                )
            if config_tags.difference(ciflow_tags):
                print(
                    "Reference in config, but not in workflows",
                    config_tags.difference(ciflow_tags),
                )
            print(f"Please run {__file__} to remediate the difference")
            sys.exit(-1)
        print("All tags are listed in pytorch-probot.yml")
    else:
        update_probot_config(pushtags)
