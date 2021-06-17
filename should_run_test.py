import argparse
import json
import subprocess
import pathlib
import tempfile
import sys
import xmltodict
import collections
import typing
import asyncio
import threading
import random
import shlex
import itertools
import multiprocessing
import re
from typing import List, Dict, Any, Tuple, Union


CHUNK_HEADER_RE = r"diff --git .*?\nindex.*?\n---.*?\n\+\+\+ b/(.*?)\n@@ -(\d+,\d+) \+(\d+,\d+) @@"


class Test(typing.NamedTuple):
    file: str
    classname: str
    test: str


def list_tests(file: str):
    proc = subprocess.run(["pytest", "--disable-warnings", "-q", "--collect-only", file], stdout=subprocess.PIPE)
    stdout = proc.stdout.decode()

    tests = []
    for line in stdout.split("\n"):
        if "::" not in line:
            continue

        line = line.strip()
        file, classname, test = line.split("::")

        tests.append(Test(file=file, classname=classname, test=test))

    return tests


def find_changed_lines(diff: str) -> Dict[str, List[Tuple[int, int]]]:
    files = collections.defaultdict(list)

    matches = re.findall(CHUNK_HEADER_RE, diff, re.MULTILINE)
    for file, start, end in matches:
        start_line, _ = start.split(",")
        end_line, _ = end.split(",")

        files[file].append((int(start_line), int(end_line)))

    return dict(files)


def match(test_lines: Dict[str, Union[int, List[int]]], pr_lines: Dict[str, List[Tuple[int, int]]]):
    tl = test_lines

    # tl is a dict of files -> lines touched
    # file names are rooted in the torch/ dir

    # pr_lines is a list of test filename -> changed lines
    # file names are rooted at the repo root
    # for 

    def match_ranges(coverage_lines, changed_lines):
        return True

    # algo is: loop over diff files (since it's way smaller), check test lines to see
    # if any of them are present (check for file name, then line ranges). if not, skip
    # the test.
    def clean_filename(name):
        if name.startswith("torch/"):
            return name[len("torch/"):]

    # for k in test_lines.keys():
    #     print(k)

    should_skip = True
    for pr_filename, changed_lines in pr_lines.items():
        # changed_lines is a list of line ranges ((int, int) tuples)
        cleaned = clean_filename(pr_filename)
        
        # print(cleaned)
        if cleaned in test_lines:
            coverage_lines = test_lines[cleaned]
            if match_ranges(coverage_lines, changed_lines):
                should_skip = False

    return should_skip


def main(args):
    with open(args.coverage, "r") as f:
        coverage = json.load(f)

    with open(args.diff, "r") as f:
        changes_lines = find_changed_lines(f.read())
    
    skip_tests = []
    not_skipped = []

    for test_file, file_data in coverage.items():
        for test_class, class_data in file_data.items():
            for test_name, test_data in class_data.items():
                if match(test_data, changes_lines):
                    skip_tests.append(Test(file=test_file, classname=test_class, test=test_name))
                else:
                    not_skipped.append(Test(file=test_file, classname=test_class, test=test_name))

    for skip in skip_tests:
        print(f"Skipped {skip.classname}.{skip.test}")

    for not_skip in not_skipped:
        print(f"NOT Skipped {not_skip.classname}.{not_skip.test}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="output skipped tests")
    parser.add_argument("--coverage", help="coverage json", required=True)
    parser.add_argument("--diff", help="patch file to use for checking line numbers", required=True)
    args = parser.parse_args()

    main(args)