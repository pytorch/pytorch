"""Identifier formats and derived names shared by every Auto PR Triage stage."""

from __future__ import annotations

import re


ACCOUNT_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
# ghstack opens each PR in a stack against its own gh/<user>/<N>/base branch.
TARGET_BASE_REF_RE = re.compile(rf"main|gh/{ACCOUNT_PATTERN}/[1-9][0-9]*/base")
TEAM_SLUG_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?"
OWNER_HANDLE_PATTERN = rf"(?:@{ACCOUNT_PATTERN}|@{ACCOUNT_PATTERN}/{TEAM_SLUG_PATTERN})"
TEAM_OWNER_ID_PATTERN = r"[a-z][a-z0-9_-]{0,63}"
USER_HANDLE_RE = re.compile(rf"@{ACCOUNT_PATTERN}")
TEAM_OWNER_ID_RE = re.compile(TEAM_OWNER_ID_PATTERN)
# Codepath owners come from CODEOWNERS, which names only GitHub users and teams.
CODEPATH_OWNER_RE = re.compile(OWNER_HANDLE_PATTERN)
