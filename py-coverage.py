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


def xml_to_per_file_lines_executed(content):
    x = xmltodict.parse(content)
    # json round trip to use standard types
    x = json.loads(json.dumps(x))
    data = x["coverage"]["packages"]["package"]

    def get_lines(lines):
        result = set()

        if isinstance(lines, dict):
            result.add(int(lines["@number"]))
        else:
            for line in lines:
                result.add(int(line["@number"]))

        return result

    files = collections.defaultdict(set)
    for item in data:
        classes = item["classes"]["class"]
        if isinstance(classes, list):
            for the_class in classes:
                name = the_class["@filename"]
                if the_class["lines"] is None:
                    # no lines executed in file
                    continue

                lines = the_class["lines"]["line"]
                files[name].update(get_lines(lines))

                # if isinstance(lines, dict):
                #     files[name].add(lines["@number"])
                # else:
                #     for line in lines:
                #         files[name].add(line["@number"])
        else:
            name = classes["@filename"]
            if classes["lines"] is None:
                # no lines in file
                continue
            lines = classes["lines"]["line"]
            files[name].update(get_lines(lines))

    return dict(files)



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


async def get_coverage_data(test: Test, try_count=0):
    test_name = f"{test.file}::{test.classname}::{test.test}"
    with tempfile.NamedTemporaryFile() as f:
        cmd = shlex.join([
            "pytest",
            "--cov=torch",
            "--disable-warnings",
            f"--cov-report=xml:{f.name}",
            test_name,
        ])
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        with open(f.name, "r") as f_r:
            content = f_r.read()
            if try_count < 2 and content.strip() == "":
                # retry the test, no coverage data for some reason
                print(try_count, "failed", cmd)
                return await get_coverage_data(test, try_count + 1)
            try:
                lines = xml_to_per_file_lines_executed(content)
            except Exception as e:
                print(cmd)
                print(content)
                raise e
            new_lines = {}

            for filename, lines_set in lines.items():
                new_lines[filename] = list(ranges(sorted(list(lines_set))))
            
            lines = new_lines
    
    # print(json.dumps(lines))
    return lines


async def gather_with_concurrency(n, tasks):
    semaphore = asyncio.Semaphore(n)
 
    async def sem_task(task):
        async with semaphore:
            await task

    return await asyncio.gather(*(sem_task(task) for task in tasks), return_exceptions=False)


async def main(args):
    all_results = {}

    finished = 0
    total = 0

    async def per_test_coverage_task(test):
        nonlocal finished
        lines = await get_coverage_data(test)

        if test.file not in all_results:
            all_results[test.file] = {}
        if test.classname not in all_results[test.file]:
            all_results[test.file][test.classname] = {}
        
        all_results[test.file][test.classname][test.test] = lines

        finished += 1
        if finished % 2 == 0:
            print(f"finished {finished} / {total}")

    tests = list_tests(args.file)
    tests = tests[:100]
    coros = [per_test_coverage_task(test) for test in tests]
    total = len(coros)

    print(f"Running {len(coros)} tests")
    results = await gather_with_concurrency(multiprocessing.cpu_count(), coros)
    for index, item in enumerate(results):
        if item is not None:
            print(index, tests[index], item)


    print("WRITING to", args.out)
    with open(args.out, "w") as f:
        json.dump(all_results, f)


def ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        if b[0][1] == b[-1][1]:
            yield b[0][1]
        else:
            yield [b[0][1], b[-1][1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run pytest-cov on some test")
    parser.add_argument("--file", help="test file", required=True)
    parser.add_argument("--out", help="output file", required=True)
    args = parser.parse_args()

    asyncio.run(main(args))

    # r = list(ranges(x))
    # print(r)