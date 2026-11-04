"""Machine-readable status of mergebot's replies to merge and revert commands.

A reply that reports how a merge or revert command went ends with a hidden tag,
for example ``<!-- mergebot-status: merge failure comment=123 -->``. GitHub
renders it as nothing and leaves it out of GraphQL ``bodyText``, so the visible
text and the text-matching consumers stay as they were. Tools such as ShipIt's
DiffTrain merge job read the newest tagged reply to their own command instead of
guessing from prose. Only comments authored by the mergebot account count: anyone
can post a comment that ends in a tag, which renders as nothing, and the parsers
here do no author check, so a reader must check the author itself.

``comment=`` is the id of the comment that issued the command, so a reader can
tell the reply to its command from the reply to an earlier one: a second merge
command cancels the first command's run, whose "canceled" reply then lands after
the second command. A revert dispatched without a comment has no ``comment=``.

``success`` means the command was accepted and is running or done ("Merge
started", "successfully started a revert job", "has been successfully
reverted", and "was merged ... but it is still open", posted when a merge landed
but GitHub left the PR open). Otherwise a merge that lands posts no reply of its
own, so the PR closing, not a tag, is the sign that it landed. ``failure`` means
the command was refused or failed before the merge or revert landed. An error after
it landed, in a follow-up step such as labeling, is reported in an untagged
reply, so the newest tag stays ``success``. The exception is a job canceled or
timed out after the merge or revert landed: its reply can't tell and says
``failure``. A later reply supersedes an earlier one: a merge that started and
then failed carries a ``success`` reply followed by a ``failure`` one.

Only replies on the PR the command was issued on are tagged. PRs merged or
reverted along with it, as part of its stack, get untagged notices; the outcome
is reported on the PR that was asked. Replies that report no outcome (notices
such as the auto-added label, waiting messages, rebase results) carry no tag.
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

    def tag(self, comment_id: int | None = None) -> str:
        # An id that isn't one is left out rather than raising, which would lose
        # the reply itself
        valid = comment_id is not None and 0 < comment_id <= MAX_COMMENT_ID
        suffix = f" comment={comment_id}" if valid else ""
        return (
            f"<!-- mergebot-status: {self.command.value} {self.result.value}"
            f"{suffix} -->"
        )


class Tag(NamedTuple):
    status: Status
    # The id of the comment that issued the command, if it was issued by one
    comment_id: int | None


MERGE_SUCCESS = Status(Command.MERGE, Result.SUCCESS)
MERGE_FAILURE = Status(Command.MERGE, Result.FAILURE)
REVERT_SUCCESS = Status(Command.REVERT, Result.SUCCESS)
REVERT_FAILURE = Status(Command.REVERT, Result.FAILURE)

# GitHub comment ids are positive 64-bit integers. The bound also keeps int()
# clear of Python's limit on converting very long digit strings
MAX_COMMENT_ID = 2**63 - 1
COMMENT_ID_RE = r"[1-9][0-9]{0,18}"

STATUS_TAG_RE = re.compile(
    r"\n<!-- mergebot-status: (?P<command>merge|revert) (?P<result>success|failure)"
    rf"(?: comment=(?P<comment_id>{COMMENT_ID_RE}))? -->\s*\Z"
)


def parse_comment_id(value: str | None) -> int | None:
    """``value`` as a comment id, or None if it isn't one."""
    if value is None or not re.fullmatch(COMMENT_ID_RE, value):
        return None
    comment_id = int(value)
    return comment_id if comment_id <= MAX_COMMENT_ID else None


def with_status(body: str, status: Status | None, comment_id: int | None = None) -> str:
    """Return ``body`` with the tag for ``status`` appended; None appends nothing.
    ``comment_id`` is the comment that issued the command the reply answers.

    The tag goes last, so a reader that checks how a reply starts
    ("## Merge failed", "### Merge started") still matches it, and a code fence
    the body leaves open is closed so the tag stays hidden. Only a tag at the
    very end of a reply counts (see parse_tag), so text in ``body`` that looks
    like one (quoted in an error message, say) is left as it is, except at the
    end of a reply that gets no tag, where it is escaped.
    """
    if status is None:
        match = STATUS_TAG_RE.search(body)
        if match is not None:
            start = match.start() + 1  # past the newline, at the "<"
            body = f"{body[:start]}&lt;{body[start + 1 :]}"
        return body
    body = body.rstrip()
    fence = open_fence(body)
    if fence:
        body += f"\n{fence}"
    return f"{body}\n\n{status.tag(comment_id)}\n"


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


def parse_tag(body: str) -> Tag | None:
    """The tag a reply carries, or None for a reply without one. Only a tag on the
    reply's last line counts. This does not check who posted the reply; the
    caller must (see the module docstring)."""
    match = STATUS_TAG_RE.search(body)
    if match is None:
        return None
    comment_id = None
    if match["comment_id"] is not None:
        comment_id = parse_comment_id(match["comment_id"])
        if comment_id is None:
            return None
    return Tag(Status(Command(match["command"]), Result(match["result"])), comment_id)


def parse_status(body: str) -> Status | None:
    """The status a reply reports, or None for a reply without a tag. Like
    parse_tag, this does not check who posted the reply."""
    tag = parse_tag(body)
    return tag.status if tag is not None else None
