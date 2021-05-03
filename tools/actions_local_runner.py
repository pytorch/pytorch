#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
import yaml
import asyncio
import shutil
import re
from typing import List, Dict, Any, Optional


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
    return col.BOLD + the_color + str(text) + col.RESET


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


async def run_step(step: Dict[str, Any], job_name: str, files: Optional[List[str]], quiet: bool) -> bool:
    env = os.environ.copy()
    env["GITHUB_WORKSPACE"] = "/tmp"
    if files is None:
        env["LOCAL_FILES"] = ""
    else:
        env["LOCAL_FILES"] = " ".join(files)
    script = step["run"]

    PASS = color(col.GREEN, '\N{check mark}')
    FAIL = color(col.RED, 'x')

    if quiet:
        # TODO: Either lint that GHA scripts only use 'set -eux' or make this more
        # resilient
        script = script.replace("set -eux", "set -eu")
        script = re.sub(r"^time ", "", script, flags=re.MULTILINE)
    name = f'{job_name}: {step["name"]}'

    def header(passed: bool) -> None:
        icon = PASS if passed else FAIL
        print(f"{icon} {color(col.BLUE, name)}")

    try:
        proc = await asyncio.create_subprocess_shell(
            script,
            shell=True,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable=shutil.which("bash"),
        )

        stdout_bytes, stderr_bytes = await proc.communicate()

        header(passed=proc.returncode == 0)
    except Exception as e:
        header(passed=False)
        print(e)
        return False

    stdout = stdout_bytes.decode().strip()
    stderr = stderr_bytes.decode().strip()

    if stderr != "":
        print(stderr)
    if stdout != "":
        print(stdout)

    return proc.returncode == 0


async def run_steps(steps: List[Dict[str, Any]], job_name: str, files: Optional[List[str]], quiet: bool) -> None:
    coros = [run_step(step, job_name, files, quiet) for step in steps]
    await asyncio.gather(*coros)


def grab_specific_steps(steps_to_grab: List[str], job: Dict[str, Any]) -> List[Dict[str, Any]]:
    relevant_steps = []
    for step in steps_to_grab:
        for actual_step in job["steps"]:
            if actual_step["name"].lower().strip() == step.lower().strip():
                relevant_steps.append(actual_step)
                break

    if len(relevant_steps) != len(steps_to_grab):
        raise RuntimeError("Missing steps")

    return relevant_steps


def grab_all_steps_after(last_step: str, job: Dict[str, Any]) -> List[Dict[str, Any]]:
    relevant_steps = []

    found = False
    for step in job["steps"]:
        if found:
            relevant_steps.append(step)
        if step["name"].lower().strip() == last_step.lower().strip():
            found = True

    return relevant_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull shell scripts out of GitHub actions and run them"
    )
    parser.add_argument("--file", help="YAML file with actions", required=True)
    parser.add_argument("--file-filter", help="only pass through files with this extension", default='')
    parser.add_argument("--changed-only", help="only run on changed files", action='store_true', default=False)
    parser.add_argument("--job", help="job name", required=True)
    parser.add_argument("--no-quiet", help="output commands", action='store_true', default=False)
    parser.add_argument("--step", action="append", help="steps to run (in order)")
    parser.add_argument(
        "--all-steps-after", help="include every step after this one (non inclusive)"
    )
    args = parser.parse_args()

    relevant_files = None
    quiet = not args.no_quiet

    if args.changed_only:
        changed_files: Optional[List[str]] = None
        try:
            changed_files = find_changed_files()
        except Exception:
            # If the git commands failed for some reason, bail out and use the whole list
            print(
                "Could not query git for changed files, falling back to testing all files instead",
                file=sys.stderr
            )
        if changed_files is not None:
            relevant_files = []
            for f in changed_files:
                for file_filter in args.file_filter:
                    if f.endswith(file_filter):
                        relevant_files.append(f)
                        break

    if args.step is None and args.all_steps_after is None:
        raise RuntimeError("1+ --steps or --all-steps-after must be provided")

    if args.step is not None and args.all_steps_after is not None:
        raise RuntimeError("Only one of --step and --all-steps-after can be used")

    action = yaml.safe_load(open(args.file, "r"))
    if "jobs" not in action:
        raise RuntimeError(f"top level key 'jobs' not found in {args.file}")
    jobs = action["jobs"]

    if args.job not in jobs:
        raise RuntimeError(f"job '{args.job}' not found in {args.file}")

    job = jobs[args.job]

    if args.step is not None:
        relevant_steps = grab_specific_steps(args.step, job)
    else:
        relevant_steps = grab_all_steps_after(args.all_steps_after, job)

    if sys.version_info > (3, 7):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_steps(relevant_steps, args.job, relevant_files, quiet))
    else:
        raise RuntimeError("Only Python >3.7 is supported")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
