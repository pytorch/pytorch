import collections
import fnmatch
import re
from typing import Pattern, List, Tuple

from tools.linter.utils import run_cmd, glob2regex


def split_positive_from_negative(patterns: List[str]):
    positive, negative = [], []

    for pattern in patterns:
        if pattern.startswith("-"):
            negative.append(pattern[1:])
        else:
            positive.append(pattern)
    return positive, negative



def split_patterns(glob: List[str], regex: List[str]) -> Tuple[List[Pattern], List[Pattern]]:
    positive_glob, negative_glob = split_positive_from_negative(glob)
    positive_regex, negative_regex = split_positive_from_negative(regex)

    positive_patterns = positive_regex + [glob2regex(g) for g in positive_glob]
    negative_patterns = negative_regex + [glob2regex(g) for g in negative_glob]

    positive = [ re.compile(p) for p in positive_patterns ]
    negative = [ re.compile(p) for p in negative_patterns ]

    return positive, negative


async def get_files_tracked_by_git(files):
    files = [file.rstrip("/") for file in files]
    result = await run_cmd(["git", "ls-files"] + files)
    return result.stdout.strip().splitlines()


async def filter_files(files: List[str], glob: List[str], regex: List[str]) -> List[str]:
    files = await get_files_tracked_by_git(files)
    positive_patterns, negative_patterns = split_patterns(glob, regex)

    return [
        file
        for file in files
        if not any(n.match(file) for n in negative_patterns)
        and any(p.match(file) for p in positive_patterns)
    ]
