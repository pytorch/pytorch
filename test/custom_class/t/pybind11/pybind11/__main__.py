from __future__ import print_function

import argparse
import sys
import sysconfig

from . import get_include


def print_includes():
    dirs = [sysconfig.get_path('include'),
            sysconfig.get_path('platinclude'),
            get_include(),
            get_include(True)]

    # Make unique but preserve order
    unique_dirs = []
    for d in dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)

    print(' '.join('-I' + d for d in unique_dirs))


def main():
    parser = argparse.ArgumentParser(prog='python -m pybind11')
    parser.add_argument('--includes', action='store_true',
                        help='Include flags for both pybind11 and Python headers.')
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.includes:
        print_includes()


if __name__ == '__main__':
    main()
