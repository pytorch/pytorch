#!/usr/bin/env python3

import fileinput
import os
import sys

NEWLINE, = b'\n'


def correct_trailing_newlines(filename: str) -> bool:
    with open(filename, 'rb') as f:
        a = len(f.read(2))
        if a == 0:
            return True
        elif a == 1:
            # file is wrong whether or not the only byte is a newline
            return False
        else:
            f.seek(-2, os.SEEK_END)
            b, c = f.read(2)
            # no ASCII byte is part of any non-ASCII character in UTF-8
            return b != NEWLINE and c == NEWLINE


def main() -> int:
    # mimic git grep exit code behavior
    exit_code = 1
    for line in fileinput.input():
        stripped = line.rstrip()
        if not correct_trailing_newlines(stripped):
            exit_code = 0
            print(stripped)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
