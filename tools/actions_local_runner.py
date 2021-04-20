#!/bin/python3

import subprocess
import os
import argparse
import yaml
import asyncio


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


class col:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color(the_color, text):
    return col.BOLD + the_color + str(text) + col.RESET


def cprint(the_color, text):
    print(color(the_color, text))


async def run_step(step, job_name):
    env = os.environ.copy()
    env["GITHUB_WORKSPACE"] = "/tmp"
    script = step["run"]

    # We don't need to print the commands for local running
    # TODO: Either lint that GHA scripts only use 'set -eux' or make this more
    # resilient
    script = script.replace("set -eux", "set -eu")

    try:
        proc = await asyncio.create_subprocess_shell(
            script,
            shell=True,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        cprint(col.BLUE, f'{job_name}: {step["name"]}')
    except Exception as e:
        cprint(col.BLUE, f'{job_name}: {step["name"]}')
        print(e)

    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()

    if stderr != "":
        print(stderr)
    if stdout != "":
        print(stdout)


async def run_steps(steps, job_name):
    coros = [run_step(step, job_name) for step in steps]
    await asyncio.gather(*coros)


def grab_specific_steps(steps_to_grab, job):
    relevant_steps = []
    for step in steps_to_grab:
        for actual_step in job["steps"]:
            if actual_step["name"].lower().strip() == step.lower().strip():
                relevant_steps.append(actual_step)
                break

    if len(relevant_steps) != len(steps_to_grab):
        raise RuntimeError("Missing steps")

    return relevant_steps


def grab_all_steps_after(last_step, job):
    relevant_steps = []

    found = False
    for step in job["steps"]:
        if found:
            relevant_steps.append(step)
        if step["name"].lower().strip() == last_step.lower().strip():
            found = True

    return relevant_steps


def main():
    parser = argparse.ArgumentParser(
        description="Pull shell scripts out of GitHub actions and run them"
    )
    parser.add_argument("--file", help="YAML file with actions", required=True)
    parser.add_argument("--job", help="job name", required=True)
    parser.add_argument("--step", action="append", help="steps to run (in order)")
    parser.add_argument(
        "--all-steps-after", help="include every step after this one (non inclusive)"
    )
    args = parser.parse_args()

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

    # pprint.pprint(relevant_steps)
    asyncio.run(run_steps(relevant_steps, args.job))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
