#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from bisect import bisect_right
from collections import defaultdict
from typing import (Callable, DefaultDict, Generic, List, Optional, Pattern,
                    Sequence, TypeVar, cast)

from typing_extensions import TypedDict


class Hunk(TypedDict):
    old_start: int
    old_count: int
    new_start: int
    new_count: int


class Diff(TypedDict):
    old_filename: Optional[str]
    hunks: List[Hunk]


# @@ -start,count +start,count @@
hunk_pattern = r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@'


def parse_diff(diff: str) -> Diff:
    name = None
    name_found = False
    hunks: List[Hunk] = []
    for line in diff.splitlines():
        hunk_match = re.match(hunk_pattern, line)
        if name_found:
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
            name_match = re.match(r'^--- (?:(?:/dev/null)|(?:a/(.*)))$', line)
            if name_match:
                name_found = True
                name, = name_match.groups()
    return {
        'old_filename': name,
        'hunks': hunks,
    }


T = TypeVar('T')
U = TypeVar('U')


# we want to use bisect.bisect_right to find the closest hunk to a given
# line number, but the bisect module won't have a key function until
# Python 3.10 https://github.com/python/cpython/pull/20556 so we make an
# O(1) wrapper around the list of hunks that makes it pretend to just be
# a list of line numbers
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


def parse_annotation(regex: Pattern[str], line: str) -> Optional[Annotation]:
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


def translate_all(
    *,
    lines: List[str],
    regex: Pattern[str],
    commit: str
) -> List[Annotation]:
    ann_dict: DefaultDict[str, List[Annotation]] = defaultdict(list)
    for line in lines:
        annotation = parse_annotation(regex, line)
        if annotation is not None:
            ann_dict[annotation['filename']].append(annotation)
    ann_list = []
    for filename, annotations in ann_dict.items():
        raw_diff = subprocess.check_output(
            ['git', 'diff-index', '--unified=0', commit, filename],
            encoding='utf-8',
        )
        diff = parse_diff(raw_diff) if raw_diff.strip() else None
        # if there is a diff but it doesn't list an old filename, that
        # means the file is absent in the commit we're targeting, so we
        # skip it
        if not (diff and not diff['old_filename']):
            for annotation in annotations:
                line_number: Optional[int] = annotation['lineNumber']
                if diff:
                    annotation['filename'] = cast(str, diff['old_filename'])
                    line_number = translate(diff, cast(int, line_number))
                if line_number:
                    annotation['lineNumber'] = line_number
                    ann_list.append(annotation)
    return ann_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--regex')
    parser.add_argument('--commit')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        lines = f.readlines()
    print(json.dumps(translate_all(
        lines=lines,
        regex=args.regex,
        commit=args.commit
    )))


if __name__ == '__main__':
    main()
