import re
import collections
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


def split_patterns(
    glob: List[str], regex: List[str]
) -> Tuple[List[Pattern], List[Pattern]]:
    positive_glob, negative_glob = split_positive_from_negative(glob)
    positive_regex, negative_regex = split_positive_from_negative(regex)

    positive_patterns = positive_regex + [glob2regex(g) for g in positive_glob]
    negative_patterns = negative_regex + [glob2regex(g) for g in negative_glob]

    positive = [re.compile(p) for p in positive_patterns]
    negative = [re.compile(p) for p in negative_patterns]

    return positive, negative


async def get_files_tracked_by_git(files):
    files = [file.rstrip("/") for file in files]
    result = await run_cmd(["git", "ls-files"] + files)
    return result.stdout.strip().splitlines()


async def filter_files(
    files: List[str], glob: List[str], regex: List[str]
) -> List[str]:
    files = await get_files_tracked_by_git(files)
    positive_patterns, negative_patterns = split_patterns(glob, regex)

    return [
        file
        for file in files
        if not any(n.match(file) for n in negative_patterns)
        and any(p.match(file) for p in positive_patterns)
    ]


def map_files_to_line_filters(diff):
    # Delay import since this isn't required unless using the --diff-file
    # argument, which for local runs people don't care about
    try:
        import unidiff  # type: ignore[import]
    except ImportError as e:
        e.msg += ", run 'pip install unidiff'"  # type: ignore[attr-defined]
        raise e

    files: Any = collections.defaultdict(list)

    for file in unidiff.PatchSet(diff):
        for hunk in file:
            added_line_nos = [line.target_line_no for line in hunk if line.is_added]

            if len(added_line_nos) == 0:
                continue

            # Convert list of line numbers to ranges
            # Eg: [1, 2, 3, 12, 13, 14, 15] becomes [[1,3], [12, 15]]
            i = 1
            ranges = [[added_line_nos[0], added_line_nos[0]]]
            while i < len(added_line_nos):
                if added_line_nos[i] != added_line_nos[i - 1] + 1:
                    ranges[-1][1] = added_line_nos[i - 1]
                    ranges.append([added_line_nos[i], added_line_nos[i]])
                i += 1
            ranges[-1][1] = added_line_nos[-1]

            files[file.path].append(*ranges)

    return dict(files)


def filter_files_from_diff(paths, diff):
    files = []
    line_filters = []
    changed_files = map_files_to_line_filters(diff)
    changed_files = {
        filename: v
        for filename, v in changed_files.items()
        if any(filename.startswith(path) for path in paths)
    }
    line_filters += [
        {"name": name, "lines": lines} for name, lines, in changed_files.items()
    ]
    files += list(changed_files.keys())
    return files, line_filters


def filter_files_from_diff_file(paths, diff_file):
    with open(diff_file, "r") as f:
        diff = f.read()
    return filter_files_from_diff(paths, diff)
