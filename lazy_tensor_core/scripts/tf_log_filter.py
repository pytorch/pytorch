#!/usr/bin/env python3

from __future__ import print_function

import argparse
import re
import sys


def normalize(args):
    fd = sys.stdin
    if args.input:
        fd = open(args.input)
    # 2019-04-06 02:51:26.397580: I lazy_tensor_core/csrc/forward.cpp:168]
    for line in fd:
        line.rstrip('\n')
        m = re.match(r'.*:\d+\] (.*)', line)
        if m:
            print(m.group(1))
        else:
            print(line)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', type=str)
    args = arg_parser.parse_args()
    normalize(args)
