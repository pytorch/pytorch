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

from typing import List, Dict, Any, Optional, Union, NamedTuple, Set

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
    icon = color(col.GREEN, "✓") if passed else color(col.RED, "x")
    print(f"{icon} {color(col.BLUE, job_name)}")

    for stream in streams:
        stream = stream.strip()
        if stream != "":
            print(stream)


class CommandResult(NamedTuple):
    passed: bool
    stdout: str
    stderr: str


async def shell_cmd(
    cmd: Union[str, List[str]],
    env: Optional[Dict[str, Any]] = None,
    redirect: bool = True,
) -> CommandResult:
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
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
        return CommandResult(passed, "", "")

    return CommandResult(passed, stdout.decode().strip(), stderr.decode().strip())


class Check:
    name: str

    def __init__(self, files: Optional[List[str]], quiet: bool):
        self.quiet = quiet
        self.files = files

    async def run(self) -> bool:
        result = await self.run_helper()
        if result is None:
            return True

        streams = []
        if not result.passed:
            streams = [
                result.stderr,
                result.stdout,
            ]
        print_results(self.name, result.passed, streams)
        return result.passed

    async def run_helper(self) -> Optional[CommandResult]:
        if self.files is not None:
            relevant_files = self.filter_files(self.files)
            if len(relevant_files) == 0:
                # No files, do nothing
                return CommandResult(passed=True, stdout="", stderr="")

            return await self.quick(relevant_files)

        return await self.full()

    def filter_ext(self, files: List[str], extensions: Set[str]) -> List[str]:
        def passes(filename: str) -> bool:
            return os.path.splitext(filename)[1] in extensions

        return [f for f in files if passes(f)]

    def filter_files(self, files: List[str]) -> List[str]:
        return files

    async def quick(self, files: List[str]) -> CommandResult:
        raise NotImplementedError

    async def full(self) -> Optional[CommandResult]:
        raise NotImplementedError


class Flake8(Check):
    name = "flake8"

    def filter_files(self, files: List[str]) -> List[str]:
        config = configparser.ConfigParser()
        config.read(os.path.join(REPO_ROOT, ".flake8"))

        excludes = re.split(r",\s*", config["flake8"]["exclude"].strip())
        excludes = [e.strip() for e in excludes if e.strip() != ""]

        def should_include(name: str) -> bool:
            for exclude in excludes:
                if fnmatch.fnmatch(name, pat=exclude):
                    return False
                if name.startswith(exclude) or f"./{name}".startswith(exclude):
                    return False
            return True

        files = self.filter_ext(files, {".py"})
        return [f for f in files if should_include(f)]

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(["flake8"] + files)

    async def full(self) -> CommandResult:
        return await shell_cmd(["flake8"])


class Mypy(Check):
    name = "mypy (skipped typestub generation)"

    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".py", ".pyi"})

    def env(self) -> Dict[str, Any]:
        env = os.environ.copy()
        if should_color():
            # Secret env variable: https://github.com/python/mypy/issues/7771
            env["MYPY_FORCE_COLOR"] = "1"
        return env

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(
            [sys.executable, "tools/linter/mypy_wrapper.py"]
            + [os.path.join(REPO_ROOT, f) for f in files],
            env=self.env(),
        )

    async def full(self) -> None:
        env = self.env()
        # hackily change the name
        self.name = "mypy"

        await shell_cmd(
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

        await shell_cmd(
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


class ShellCheck(Check):
    name = "shellcheck: Run ShellCheck"

    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".sh"})

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(
            ["tools/linter/run_shellcheck.sh"]
            + [os.path.join(REPO_ROOT, f) for f in files],
        )

    async def full(self) -> None:
        await shell_cmd(
            [
                sys.executable,
                "tools/actions_local_runner.py",
                "--job",
                "shellcheck",
                "--file",
                ".github/workflows/lint.yml",
                "--step",
                "Run ShellCheck",
            ],
            redirect=False,
        )


class ClangTidy(Check):
    name = "clang-tidy: Run clang-tidy"
    common_options = [
        "--clang-tidy-exe",
        ".clang-tidy-bin/clang-tidy",
        "--parallel",
    ]

    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".c", ".cc", ".cpp"})

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(
            [sys.executable, "tools/linter/clang_tidy", "--paths"]
            + [os.path.join(REPO_ROOT, f) for f in files]
            + self.common_options,
        )

    async def full(self) -> None:
        await shell_cmd(
            [sys.executable, "tools/linter/clang_tidy"] + self.common_options
        )


class YamlStep(Check):
    def __init__(self, step: Dict[str, Any], job_name: str, quiet: bool):
        super().__init__(files=None, quiet=quiet)
        self.step = step
        self.name = f'{job_name}: {self.step["name"]}'

    async def full(self) -> CommandResult:
        env = os.environ.copy()
        env["GITHUB_WORKSPACE"] = "/tmp"
        script = self.step["run"]

        if self.quiet:
            # TODO: Either lint that GHA scripts only use 'set -eux' or make this more
            # resilient
            script = script.replace("set -eux", "set -eu")
            script = re.sub(r"^time ", "", script, flags=re.MULTILINE)

        return await shell_cmd(script, env=env)


def changed_files() -> Optional[List[str]]:
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

    return changed_files


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
        raise RuntimeError(f"Missing steps:\n{relevant_steps}\n{steps_to_grab}")

    return relevant_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull shell scripts out of GitHub actions and run them"
    )
    parser.add_argument("--file", help="YAML file with actions")
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

    quiet = not args.no_quiet

    if args.file is None:
        # If there is no .yml file provided, fall back to the list of known
        # jobs. We use this for flake8 and mypy since they run different
        # locally than in CI due to 'make quicklint'
        if args.job not in ad_hoc_steps:
            raise RuntimeError(
                f"Job {args.job} not found and no .yml file was provided"
            )

        files = None
        if args.changed_only:
            files = changed_files()

        checks = [ad_hoc_steps[args.job](files, quiet)]
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
        checks = [
            YamlStep(step=step, job_name=args.job, quiet=quiet)
            for step in relevant_steps
        ]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*[check.run() for check in checks]))


# These are run differently locally in order to enable quicklint, so dispatch
# out to special handlers instead of using lint.yml
ad_hoc_steps = {
    "mypy": Mypy,
    "flake8-py3": Flake8,
    "shellcheck": ShellCheck,
    "clang-tidy": ClangTidy,
}

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
