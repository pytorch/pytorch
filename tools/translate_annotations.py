#!/usr/bin/env python3

import argparse
import json
import re
from bisect import bisect_right
from typing import (Callable, Generic, List, Optional, Pattern, Sequence,
                    TypeVar, cast)

from typing_extensions import TypedDict


class Hunk(TypedDict):
    old_start: int
    old_count: int
    new_start: int
    new_count: int


class Diff(TypedDict):
    old_filename: str
    hunks: List[Hunk]


# adapted from the similar regex in tools/clang_tidy.py
# @@ -start,count +start,count @@
hunk_pattern = r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@'


def parse_diff(diff: str) -> Diff:
    name = None
    hunks: List[Hunk] = []
    for line in diff.splitlines():
        name_match = re.match(r'^--- a/(.*)$', line)
        hunk_match = re.match(hunk_pattern, line)
        if name:
            assert not name_match
            if hunk_match:
                old_start, old_count, new_start, new_count = hunk_match.groups()
                hunks.append({
                    'old_start': int(old_start),
                    'old_count': int(old_count or '1'),
                    'new_start': int(new_start),
                    'new_count': int(new_count or '1'),
                })
        else:
            assert not hunk_match
            if name_match:
                name, = name_match.groups()
    assert name
    return {
        'old_filename': name,
        'hunks': hunks,
    }


T = TypeVar('T')
U = TypeVar('U')


# https://gist.github.com/ericremoreynolds/2d80300dabc70eebc790
class KeyifyList(Generic[T, U]):
    def __init__(self, inner: List[T], key: Callable[[T], U]) -> None:
        self.inner = inner
        self.key = key

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, k: int) -> U:
        return self.key(self.inner[k])


def translate(diff: Diff, line_number: int) -> Optional[int]:
    if line_number < 1:
        return None

    hunks = diff['hunks']
    if not hunks:
        return line_number

    keyified = KeyifyList(
        hunks,
        lambda hunk: hunk['new_start'] + (0 if hunk['new_count'] > 0 else 1)
    )
    i = bisect_right(cast(Sequence[int], keyified), line_number)
    if i < 1:
        return line_number

    hunk = hunks[i - 1]
    d = line_number - (hunk['new_start'] + (hunk['new_count'] or 1))
    return None if d < 0 else hunk['old_start'] + (hunk['old_count'] or 1) + d


# we use camelCase here because this will be output as JSON and so the
# field names need to match the group names from here:
# https://github.com/pytorch/add-annotations-github-action/blob/3ab7d7345209f5299d53303f7aaca7d3bc09e250/action.yml#L23
class Annotation(TypedDict):
    filename: str
    lineNumber: int
    columnNumber: int
    errorCode: str
    errorDesc: str


def parse(regex: Pattern[str], line: str) -> Optional[Annotation]:
    m = re.match(regex, line)
    if m:
        try:
            line_number = int(m.group('lineNumber'))
            column_number = int(m.group('columnNumber'))
        except ValueError:
            return None
        return {
            'filename': m.group('filename'),
            'lineNumber': line_number,
            'columnNumber': column_number,
            'errorCode': m.group('errorCode'),
            'errorDesc': m.group('errorDesc'),
        }
    else:
        return None


def translate_all(regex: Pattern[str], lines: List[str]) -> List[Annotation]:
    annotations = []
    for line in lines:
        annotation = parse(regex, line)
        if annotation:
            annotations.append(annotation)
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--regex')
    parser.add_argument('--commit')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        lines = f.readlines()
    print(json.dumps(translate_all(args.regex, lines)))


if __name__ == '__main__':
    main()
