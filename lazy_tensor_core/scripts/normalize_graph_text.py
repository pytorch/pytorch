#!/usr/bin/env python3

from __future__ import print_function

import argparse
import re
import sys


def normalize(args):
    fd = sys.stdin
    if args.input:
        fd = open(args.input)
    for line in fd:
        line.rstrip('\n')
        m = re.match(r'(\s*)%\d+\s*=\s*(.*::[^(]+\()[^)]*(.*)', line)
        if m:
            line = m.group(1) + m.group(2) + m.group(3)
        print(line)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', type=str)
    args = arg_parser.parse_args()
    normalize(args)
