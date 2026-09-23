"""One way to compare GitHub logins across the greenlight guard, and who greenlight is.

The guard reads logins from two GitHub APIs that spell the same account differently:
GraphQL renders a GitHub App's login bare while REST appends `[bot]`. Every comparison
has to go through ``normalize_login`` or the two spellings silently fail to match, which
here means either the guard switches itself off or it refuses a merge it should allow.
"""

from __future__ import annotations


GREENLIGHT_LOGIN = "pytorchgreenlight"

# A GitHub App acts through an account whose REST login is `<app-slug>[bot]`; mirrors
# greenlight's constants.BOT_LOGIN_SUFFIX in pytorch/test-infra under
# `greenlight/src/greenlight/`.
BOT_LOGIN_SUFFIX = "[bot]"

# Mirrors greenlight's pr_hash.BOT_LOGINS, which is what decides whose review counts as
# a human's. The `[bot]` suffix alone does not identify a bot, because GraphQL strips it.
BOT_LOGINS = frozenset(
    {
        "codecov",
        "codecov-commenter",
        "dependabot",
        "facebook-github-bot",
        "facebook-github-tools",
        "github-actions",
        "linux-foundation-easycla",
        "meta-codesync",
        "pytorch-bot",
        "pytorchbot",
        "pytorchmergebot",
        "pytorchupdatebot",
    }
)


def normalize_login(login: str) -> str:
    """A login in the single form every comparison in the guard uses."""
    return login.strip().lower().removesuffix(BOT_LOGIN_SUFFIX)


def is_greenlight(login: str) -> bool:
    return normalize_login(login) == GREENLIGHT_LOGIN


def is_known_bot(login: str) -> bool:
    """Whether this login belongs to a bot, by the same test greenlight applies."""
    normalized = login.strip().lower()
    return normalized.endswith(BOT_LOGIN_SUFFIX) or normalized in BOT_LOGINS
