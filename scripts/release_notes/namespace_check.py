import argparse
import json
from os import path

import torch

# Import all utils so that getattr below can find them
from torch.utils import bottleneck, checkpoint, model_zoo

all_submod_list = [
    "",
    "nn",
    "nn.functional",
    "nn.init",
    "optim",
    "autograd",
    "cuda",
    "sparse",
    "distributions",
    "fft",
    "linalg",
    "jit",
    "distributed",
    "futures",
    "onnx",
    "random",
    "utils.bottleneck",
    "utils.checkpoint",
    "utils.data",
    "utils.model_zoo",
]


def get_content(submod):
    mod = torch
    if submod:
        submod = submod.split(".")
        for name in submod:
            mod = getattr(mod, name)
    content = dir(mod)
    return content


def namespace_filter(data):
    out = {d for d in data if d[0] != "_"}
    return out


def run(args, submod):
    print(f"## Processing torch.{submod}")
    prev_filename = f"prev_data_{submod}.json"
    new_filename = f"new_data_{submod}.json"

    if args.prev_version:
        content = get_content(submod)
        with open(prev_filename, "w") as f:
            json.dump(content, f)
        print("Data saved for previous version.")
    elif args.new_version:
        content = get_content(submod)
        with open(new_filename, "w") as f:
            json.dump(content, f)
        print("Data saved for new version.")
    else:
        assert args.compare
        if not path.exists(prev_filename):
            raise RuntimeError("Previous version data not collected")

        if not path.exists(new_filename):
            raise RuntimeError("New version data not collected")

        with open(prev_filename, "r") as f:
            prev_content = set(json.load(f))

        with open(new_filename, "r") as f:
            new_content = set(json.load(f))

        if not args.show_all:
            prev_content = namespace_filter(prev_content)
            new_content = namespace_filter(new_content)

        if new_content == prev_content:
            print("Nothing changed.")
            print("")
        else:
            print("Things that were added:")
            print(new_content - prev_content)
            print("")

            print("Things that were removed:")
            print(prev_content - new_content)
            print("")


def main():
    parser = argparse.ArgumentParser(
        description="Tool to check namespace content changes"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prev-version", action="store_true")
    group.add_argument("--new-version", action="store_true")
    group.add_argument("--compare", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submod", default="", help="part of the submodule to check")
    group.add_argument(
        "--all-submod",
        action="store_true",
        help="collects data for all main submodules",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="show all the diff, not just public APIs",
    )

    args = parser.parse_args()

    if args.all_submod:
        submods = all_submod_list
    else:
        submods = [args.submod]

    for mod in submods:
        run(args, mod)


if __name__ == "__main__":
    main()
