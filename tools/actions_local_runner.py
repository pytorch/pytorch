#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import argparse
import yaml
import asyncio
import shutil
import re
import fnmatch
import shlex
import configparser

from typing import List, Dict, Any, Optional, Tuple, Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class col:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def should_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def color(the_color: str, text: str) -> str:
    if should_color():
        return col.BOLD + the_color + str(text) + col.RESET
    else:
        return text


def cprint(the_color: str, text: str) -> None:
    if should_color():
        print(color(the_color, text))
    else:
        print(text)


def git(args: List[str]) -> List[str]:
    p = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = p.stdout.decode().strip().split("\n")
    return [line.strip() for line in lines]


def find_changed_files() -> List[str]:
    untracked = []

    for line in git(["status", "--porcelain"]):
        # Untracked files start with ??, so grab all of those
        if line.startswith("?? "):
            untracked.append(line.replace("?? ", ""))

    # Modified, unstaged
    modified = git(["diff", "--name-only"])

    # Modified, staged
    cached = git(["diff", "--cached", "--name-only"])

    # Committed
    merge_base = git(["merge-base", "origin/master", "HEAD"])[0]
    diff_with_origin = git(["diff", "--name-only", merge_base, "HEAD"])

    # De-duplicate
    all_files = set()
    for x in untracked + cached + modified + diff_with_origin:
        stripped = x.strip()
        if stripped != "" and os.path.exists(stripped):
            all_files.add(stripped)
    return list(all_files)


def print_results(job_name: str, passed: bool, streams: List[str]) -> None:
    header(job_name, passed)
    for stream in streams:
        stream = stream.strip()
        if stream != "":
            print(stream)


async def shell_cmd(
    cmd: Union[str, List[str]],
    env: Optional[Dict[str, Any]] = None,
    redirect: bool = True,
) -> Tuple[bool, str, str]:
    if isinstance(cmd, list):
        cmd_str = shlex.join(cmd)  # type: ignore[attr-defined]
    else:
        cmd_str = cmd

    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        shell=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE if redirect else None,
        stderr=subprocess.PIPE if redirect else None,
        executable=shutil.which("bash"),
    )
    stdout, stderr = await proc.communicate()

    passed = proc.returncode == 0
    if not redirect:
        return passed, "", ""

    return passed, stdout.decode().strip(), stderr.decode().strip()


def header(name: str, passed: bool) -> None:
    PASS = color(col.GREEN, "✓")
    FAIL = color(col.RED, "x")
    icon = PASS if passed else FAIL
    print(f"{icon} {color(col.BLUE, name)}")


def get_flake_excludes() -> List[str]:
    config = configparser.ConfigParser()
    config.read(os.path.join(REPO_ROOT, ".flake8"))

    excludes = re.split(r',\s*', config["flake8"]["exclude"].strip())
    excludes = [e.strip() for e in excludes if e.strip() != ""]
    return excludes


async def run_flake8(files: Optional[List[str]], quiet: bool) -> bool:
    cmd = ["flake8"]

    excludes = get_flake_excludes()

    def should_include(name: str) -> bool:
        for exclude in excludes:
            if fnmatch.fnmatch(name, pat=exclude):
                return False
            if name.startswith(exclude) or ("./" + name).startswith(exclude):
                return False
        return True

    if files is not None:
        files = [f for f in files if should_include(f)]

        if len(files) == 0:
            print_results("flake8", True, [])
            return True

        # Running quicklint, pass in an explicit list of files (unlike mypy,
        # flake8 will still use .flake8 to filter this list by the 'exclude's
        # in the config
        cmd += files

    passed, stdout, stderr = await shell_cmd(cmd)
    print_results("flake8", passed, [stdout, stderr])
    return passed


async def run_mypy(files: Optional[List[str]], quiet: bool) -> bool:
    env = os.environ.copy()
    if should_color():
        # Secret env variable: https://github.com/python/mypy/issues/7771
        env["MYPY_FORCE_COLOR"] = "1"

    if files is not None:
        # Running quick lint, use mypy-wrapper instead so it checks that the files
        # actually should be linted

        passed, stdout, stderr = await shell_cmd(
            [sys.executable, "tools/mypy_wrapper.py"] + [
                os.path.join(REPO_ROOT, f) for f in files
            ],
            env=env,
        )

        print_results("mypy (skipped typestub generation)", passed, [
            stdout + "\n",
            stderr + "\n",
        ])
        return passed

    # Not running quicklint, so use lint.yml
    _, _, _ = await shell_cmd(
        [
            sys.executable,
            "tools/actions_local_runner.py",
            "--job",
            "mypy",
            "--file",
            ".github/workflows/lint.yml",
            "--step",
            "Run autogen",
        ],
        redirect=False,
        env=env,
    )
    passed, _, _ = await shell_cmd(
        [
            sys.executable,
            "tools/actions_local_runner.py",
            "--job",
            "mypy",
            "--file",
            ".github/workflows/lint.yml",
            "--step",
            "Run mypy",
        ],
        redirect=False,
        env=env,
    )
    return passed


async def run_step(
    step: Dict[str, Any], job_name: str, files: Optional[List[str]], quiet: bool
) -> bool:
    env = os.environ.copy()
    env["GITHUB_WORKSPACE"] = "/tmp"
    script = step["run"]

    if quiet:
        # TODO: Either lint that GHA scripts only use 'set -eux' or make this more
        # resilient
        script = script.replace("set -eux", "set -eu")
        script = re.sub(r"^time ", "", script, flags=re.MULTILINE)
    name = f'{job_name}: {step["name"]}'

    passed, stderr, stdout = await shell_cmd(script, env=env)
    print_results(name, passed, [stdout, stderr])

    return passed


async def run_steps(
    steps: List[Dict[str, Any]], job_name: str, files: Optional[List[str]], quiet: bool
) -> bool:
    coros = [run_step(step, job_name, files, quiet) for step in steps]
    return all(await asyncio.gather(*coros))


def relevant_changed_files(file_filters: Optional[List[str]]) -> Optional[List[str]]:
    changed_files: Optional[List[str]] = None
    try:
        changed_files = sorted(find_changed_files())
    except Exception:
        # If the git commands failed for some reason, bail out and use the whole list
        print(
            "Could not query git for changed files, falling back to testing all files instead",
            file=sys.stderr,
        )
        return None

    if file_filters is None:
        return changed_files
    else:
        relevant_files = []
        for f in changed_files:
            for file_filter in file_filters:
                if f.endswith(file_filter):
                    relevant_files.append(f)
                    break
        return relevant_files


def grab_specific_steps(
    steps_to_grab: List[str], job: Dict[str, Any]
) -> List[Dict[str, Any]]:
    relevant_steps = []
    for step in steps_to_grab:
        for actual_step in job["steps"]:
            if actual_step["name"].lower().strip() == step.lower().strip():
                relevant_steps.append(actual_step)
                break

    if len(relevant_steps) != len(steps_to_grab):
        raise RuntimeError("Missing steps")

    return relevant_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull shell scripts out of GitHub actions and run them"
    )
    parser.add_argument("--file", help="YAML file with actions")
    parser.add_argument(
        "--file-filter",
        help="only pass through files with this extension",
        nargs="*",
    )
    parser.add_argument(
        "--changed-only",
        help="only run on changed files",
        action="store_true",
        default=False,
    )
    parser.add_argument("--job", help="job name", required=True)
    parser.add_argument(
        "--no-quiet", help="output commands", action="store_true", default=False
    )
    parser.add_argument("--step", action="append", help="steps to run (in order)")
    args = parser.parse_args()

    relevant_files = None
    quiet = not args.no_quiet

    if args.changed_only:
        relevant_files = relevant_changed_files(args.file_filter)

    if args.file is None:
        # If there is no .yml file provided, fall back to the list of known
        # jobs. We use this for flake8 and mypy since they run different
        # locally than in CI due to 'make quicklint'
        if args.job not in ad_hoc_steps:
            raise RuntimeError(
                f"Job {args.job} not found and no .yml file was provided"
            )
        future = ad_hoc_steps[args.job](relevant_files, quiet)
    else:
        if args.step is None:
            raise RuntimeError("1+ --steps must be provided")

        action = yaml.safe_load(open(args.file, "r"))
        if "jobs" not in action:
            raise RuntimeError(f"top level key 'jobs' not found in {args.file}")
        jobs = action["jobs"]

        if args.job not in jobs:
            raise RuntimeError(f"job '{args.job}' not found in {args.file}")

        job = jobs[args.job]

        # Pull the relevant sections out of the provided .yml file and run them
        relevant_steps = grab_specific_steps(args.step, job)
        future = run_steps(relevant_steps, args.job, relevant_files, quiet)

    if sys.version_info >= (3, 8):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(future)
    else:
        raise RuntimeError("Only Python >=3.8 is supported")


# These are run differently locally in order to enable quicklint, so dispatch
# out to special handlers instead of using lint.yml
ad_hoc_steps = {
    "mypy": run_mypy,
    "flake8-py3": run_flake8,
}

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
