"""Machine-readable status of mergebot's replies to merge and revert commands.

A reply that reports how a merge or revert command went ends with a hidden tag,
for example ``<!-- mergebot-status: merge failure -->``. GitHub renders it as
nothing and leaves it out of GraphQL ``bodyText``, so the visible text and the
text-matching consumers stay as they were. Tools such as ShipIt's DiffTrain
merge job read the newest tagged reply instead of guessing from prose.

``success`` means the command was accepted and is running or done ("Merge
started", "successfully started a revert job", "has been successfully
reverted"). A merge that lands closes the PR and posts no reply of its own, so
the PR closing, not a tag, is the sign that it landed. ``failure`` means the
command was refused or failed. A later reply supersedes an earlier one: a merge
that started and then failed carries a ``success`` reply followed by a
``failure`` one.

Replies that report neither (notices such as the auto-added label, waiting
messages, rebase results) carry no tag.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple


class Command(str, Enum):
    MERGE = "merge"
    REVERT = "revert"


class Result(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class Status(NamedTuple):
    command: Command
    result: Result

    def tag(self) -> str:
        return f"<!-- mergebot-status: {self.command.value} {self.result.value} -->"


MERGE_SUCCESS = Status(Command.MERGE, Result.SUCCESS)
MERGE_FAILURE = Status(Command.MERGE, Result.FAILURE)
REVERT_SUCCESS = Status(Command.REVERT, Result.SUCCESS)
REVERT_FAILURE = Status(Command.REVERT, Result.FAILURE)

TAG_PREFIX = "<!-- mergebot-status:"
STATUS_TAG_RE = re.compile(
    r"\n<!-- mergebot-status: (?P<command>merge|revert) (?P<result>success|failure) -->"
    r"\s*\Z"
)


def with_status(body: str, status: Status | None) -> str:
    """Return ``body`` with the tag for ``status`` appended; None appends nothing.

    The tag goes last, so a reader that checks how a reply starts
    ("## Merge failed", "### Merge started") still matches it. Text in ``body``
    that looks like a tag (quoted in an error message, say) is escaped, so only
    the tag added here counts, and a code fence the body leaves open is closed
    so the tag stays hidden.
    """
    body = body.replace(TAG_PREFIX, "&lt;!-- mergebot-status:")
    if status is None:
        return body
    body = body.rstrip()
    fence = open_fence(body)
    if fence:
        body += f"\n{fence}"
    return f"{body}\n\n{status.tag()}\n"


FENCE_RE = re.compile(r" {0,3}(`{3,}|~{3,})")


def open_fence(body: str) -> str | None:
    """The fence that closes a code block ``body`` leaves open, if any.

    Follows Markdown's rule: a block opens with three or more backticks or
    tildes and closes with a line of the same character, at least as long.
    """
    opened: str | None = None
    for line in body.split("\n"):
        match = FENCE_RE.match(line)
        if match is None:
            continue
        marker, rest = match[1], line[match.end() :]
        if opened is None:
            # A backtick fence's info string can't contain a backtick (that
            # line is inline code, not a fence)
            if not (marker[0] == "`" and "`" in rest):
                opened = marker
        elif marker[0] == opened[0] and len(marker) >= len(opened) and not rest.strip():
            opened = None
    return opened


def parse_status(body: str) -> Status | None:
    """The status a reply reports, or None for a reply without a tag. Only a tag
    on the reply's last line counts."""
    match = STATUS_TAG_RE.search(body)
    if match is None:
        return None
    return Status(Command(match["command"]), Result(match["result"]))
