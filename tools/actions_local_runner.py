#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
import yaml
import asyncio
import shutil
import re
import shlex
import configparser

from typing import List, Dict, Any, Optional, Tuple

from mypy_wrapper import is_match  # type: ignore[import]

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


def color(the_color: str, text: str) -> str:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return col.BOLD + the_color + str(text) + col.RESET
    else:
        return text


def cprint(the_color: str, text: str) -> None:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
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
    all_files = set(untracked + cached + modified + diff_with_origin)
    return [x.strip() for x in all_files if x.strip() != ""]


def print_results(job_name: str, passed: bool, streams: List[str]) -> None:
    header(job_name, passed)
    for stream in streams:
        stream = stream.strip()
        if stream != "":
            print(stream)


async def shell_cmd(
    cmd: str, env: Optional[Dict[str, Any]] = None, redirect: bool = True
) -> Tuple[bool, str, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
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

    return passed, stdout.decode(), stderr.decode()


PASS = color(col.GREEN, "\N{check mark}")
FAIL = color(col.RED, "x")


def header(name: str, passed: bool) -> None:
    icon = PASS if passed else FAIL
    print(f"{icon} {color(col.BLUE, name)}")


def get_flake_excludes() -> List[str]:
    config = configparser.ConfigParser()
    config.read(os.path.join(REPO_ROOT, ".flake8"))

    excludes = config["flake8"]["exclude"].split("\n")
    return [i.strip() for i in excludes if i.strip() != ""]


async def run_flake8(files: Optional[List[str]], quiet: bool) -> bool:
    cmd = "flake8"
    if files is not None:
        excludes = get_flake_excludes()
        files = [
            f
            for f in files
            if not any(is_match(pattern=exclude, filename=f) for exclude in excludes)
        ]
        if len(files) == 0:
            print_results("flake8", True, [])
            return True

        # shlex.join is Python 3.8+, but we guard for that before getting here
        # so ignore mypy
        cmd = f"flake8 {shlex.join(files)}"  # type: ignore[attr-defined]

    passed, stdout, stderr = await shell_cmd(cmd)
    print_results("flake8", passed, [stdout, stderr])
    return passed


async def run_mypy(files: Optional[List[str]], quiet: bool) -> bool:
    if files is not None:
        # Running quick lint, use mypy-wrapper instead so it checks that the files
        # actually should be linted
        stdout = ""
        stderr = ""
        passed = True

        # Pass each file to the mypy_wrapper script
        # TODO: Fix mypy wrapper to mock mypy's args and take in N files instead
        # of just 1 at a time
        for f in files:
            f = os.path.join(REPO_ROOT, f)
            f_passed, f_stdout, f_stderr = await shell_cmd(
                f"{sys.executable} tools/mypy_wrapper.py '{f}'"
            )
            if not f_passed:
                passed = False
            stdout += f_stdout + "\n"
            stderr += f_stderr + "\n"

        print_results("mypy (skipped typestub generation)", passed, [stdout, stderr])
        return passed

    # Not running quicklint, so use lint.yml
    _, _, _ = await shell_cmd(
        f"{sys.executable} tools/actions_local_runner.py --job 'mypy' --file .github/workflows/lint.yml --step 'Run autogen'",
        redirect=False,
    )
    passed, _, _ = await shell_cmd(
        f"{sys.executable} tools/actions_local_runner.py --job 'mypy' --file .github/workflows/lint.yml --step 'Run mypy'",
        redirect=False,
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
        changed_files = find_changed_files()
    except Exception:
        # If the git commands failed for some reason, bail out and use the whole list
        print(
            "Could not query git for changed files, falling back to testing all files instead",
            file=sys.stderr,
        )
        return None

    if changed_files is None:
        print(
            "Did not find any changed files, falling back to testing all files instead",
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
        raise RuntimeError("Only Python >3.7 is supported")


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
