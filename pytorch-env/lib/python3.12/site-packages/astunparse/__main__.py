from __future__ import print_function
import sys
import os
import argparse
from .unparser import roundtrip
from . import dump


def roundtrip_recursive(target, dump_tree=False):
    if os.path.isfile(target):
        print(target)
        print("=" * len(target))
        if dump_tree:
            dump(target)
        else:
            roundtrip(target)
        print()
    elif os.path.isdir(target):
        for item in os.listdir(target):
            if item.endswith(".py"):
                roundtrip_recursive(os.path.join(target, item), dump_tree)
    else:
        print(
            "WARNING: skipping '%s', not a file or directory" % target,
            file=sys.stderr
        )


def main(args):
    parser = argparse.ArgumentParser(prog="astunparse")
    parser.add_argument(
        'target',
        nargs='+',
        help="Files or directories to show roundtripped source for"
    )
    parser.add_argument(
        '--dump',
        type=bool,
        help="Show a pretty-printed AST instead of the source"
    )

    arguments = parser.parse_args(args)
    for target in arguments.target:
        roundtrip_recursive(target, dump_tree=arguments.dump)


if __name__ == "__main__":
    main(sys.argv[1:])
