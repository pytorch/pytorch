import fnmatch
import re
import asyncio
import shlex
import os
import argparse
import sys
from argparse import Action
from typing import List, Union, Optional, Dict, Any


class CommandResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout.strip()
        self.stderr = stderr.strip()

    def failed(self) -> bool:
        return self.returncode != 0

    def __add__(self, other: "CommandResult") -> "CommandResult":
        return CommandResult(
            self.returncode + other.returncode,
            f"{self.stdout}\n{other.stdout}",
            f"{self.stderr}\n{other.stderr}",
        )

    def __str__(self) -> str:
        return f"{self.stdout}"

    def __repr__(self) -> str:
        return (
            f"returncode: {self.returncode}\n"
            + f"stdout:\n{indent(self.stdout, 4)}\n"
            + f"stderr:\n{indent(self.stderr, 4)}"
        )


class ProgressMeter:
    def __init__(
        self, num_items: int, start_msg: str = "", disable_progress_bar: bool = False
    ) -> None:
        self.num_items = num_items
        self.num_processed = 0
        self.width = 80
        self.disable_progress_bar = disable_progress_bar

        # helper escape sequences
        self._clear_to_end = "\x1b[2K"
        self._move_to_previous_line = "\x1b[F"
        self._move_to_start_of_line = "\r"
        self._move_to_next_line = "\n"

        if self.disable_progress_bar:
            print(start_msg)
        else:
            self._write(
                start_msg
                + self._move_to_next_line
                + "[>"
                + (self.width * " ")
                + "]"
                + self._move_to_start_of_line
            )
            self._flush()

    def _write(self, s: str) -> None:
        sys.stderr.write(s)

    def _flush(self) -> None:
        sys.stderr.flush()

    def update(self, msg: str) -> None:
        if self.disable_progress_bar:
            return

        # Once we've processed all items, clear the progress bar
        if self.num_processed == self.num_items - 1:
            self._write(
                self._move_to_start_of_line
                + self._clear_to_end
                + self._move_to_previous_line
                + self._clear_to_end
            )
            return

        # NOP if we've already processed all items
        if self.num_processed > self.num_items:
            return

        self.num_processed += 1

        self._write(
            self._move_to_previous_line
            + self._clear_to_end
            + msg
            + self._move_to_next_line
        )

        progress = int((self.num_processed / self.num_items) * self.width)
        padding = self.width - progress
        self._write(
            self._move_to_start_of_line
            + self._clear_to_end
            + f"({self.num_processed} of {self.num_items}) "
            + f"[{progress*'='}>{padding*' '}]"
            + self._move_to_start_of_line
        )
        self._flush()

    def print(self, msg: str) -> None:
        if self.disable_progress_bar:
            print(msg)
        else:
            self._write(
                self._clear_to_end
                + self._move_to_previous_line
                + self._clear_to_end
                + msg
                + self._move_to_next_line
                + self._move_to_next_line
            )
            self._flush()


class Glob2RegexAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string):
        setattr(
            namespace,
            self.dest,
            getattr(namespace, self.dest, []) + [glob2regex(value)],
        )


async def run_cmd(
    cmd: Union[str, List[str]],
    env: Optional[Dict[str, Any]] = None,
    on_completed=None,
    on_completed_args=None,
):
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    else:
        cmd_str = cmd

    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    output = await proc.communicate()

    result = CommandResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=output[0].decode("utf-8").strip(),
        stderr=output[1].decode("utf-8").strip(),
    )

    if on_completed:
        if not on_completed_args:
            on_completed_args = []
        on_completed(result, *on_completed_args)
    return result


async def git(args):
    result = await run_cmd(["git"] + args)
    return result.stdout.splitlines()


async def find_changed_files():
    untracked = []
    for line in await git(["status", "--porcelain"]):
        # Untracked files start with ??, so grab all of those
        if line.startswith("?? "):
            untracked.append(line.replace("?? ", ""))

    # Modified, unstaged
    modified = await git(["diff", "--name-only"])

    # Modified, staged
    cached = await git(["diff", "--cached", "--name-only"])

    # Committed
    merge_base = (await git(["merge-base", "origin/master", "HEAD"]))[0]
    diff_with_origin = await git(["diff", "--name-only", merge_base, "HEAD"])

    # De-duplicate
    all_files = set()
    for x in untracked + cached + modified + diff_with_origin:
        stripped = x.strip()
        if stripped != "" and os.path.exists(stripped):
            all_files.add(stripped)
    return list(all_files)


def glob2regex(s):
    return fnmatch.translate(s)


def kebab2snake(s: str) -> str:
    return "_".join(s.split("-"))


def kebab2camel(s: str) -> str:
    return "".join([w.title() for w in s.split("-")])


class Color:
    red = "\033[91m"
    green = "\033[92m"
    reset = "\033[0m"


def color(s: str, color: Color) -> str:
    return f"{color}{s}{Color.reset}"


def indent(text, amt):
    padding = amt * " "
    return "".join(padding + line for line in text.splitlines(True))
