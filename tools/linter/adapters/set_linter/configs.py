import argparse
import json


def add_configs(args: argparse.Namespace, filename: str) -> argparse.Namespace:
    try:
        with open(filename) as fp:
            config = json.load(fp)
    except FileNotFoundError:
        config = {}

    if bad := set(config) - set(vars(args)):
        s = "" if len(bad) == 1 else "s"
        bad_name = ", ".join(sorted(bad))
        raise ValueError(f"Unknown arg{s}: {bad_name}")

    for k, v in config.items():
        if k == "fix" and args.fix is None:
            args.fix = v
        else:
            setattr(args, k, getattr(args, k) or v)

    return args


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add("files", nargs="*", help="Files or directories to include")
    add("-e", "--exclude", action="append", help="Files to exclude from the check")
    add("-f", "--fix", default=None, action="store_true", help="Fix any issues")
    add("-v", "--verbose", default=None, action="store_true", help="Print more info")

    return parser


def get_args(config_filename: str) -> argparse.Namespace:
    args = make_parser().parse_args()
    add_configs(args, config_filename)
    return args
