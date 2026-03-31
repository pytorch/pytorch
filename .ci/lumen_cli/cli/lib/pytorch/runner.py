"""
Lint test runner — local execution.

Usage:
    lumen test lint --group-id lintrunner_noclang
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS, TestPlan
from cli.lib.pytorch.re_runner import submit_command


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _env_vars(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables, restoring originals on exit."""
    backup = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, orig in backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig


def _rerun_hint(
    group_id: str,
    env_overrides: dict[str, str] | None = None,
    input_overrides: dict[str, str] | None = None,
) -> str:
    parts = [f"lumen test lint --group-id {group_id}"]
    for k, v in (env_overrides or {}).items():
        parts.append(f"--env {shlex.quote(f'{k}={v}')}")
    for k, v in (input_overrides or {}).items():
        parts.append(f"--input {shlex.quote(f'{k}={v}')}")
    parts.append("--re")
    return " ".join(parts)


def run_plan(
    group_id: str,
    plan: TestPlan,
    env_overrides: dict[str, str] | None = None,
    input_overrides: dict[str, str] | None = None,
) -> None:
    """Run a test plan locally."""
    logger.info("[%s] %s", group_id, plan.title)

    resolved = plan.resolve_env_vars(env_overrides)
    with _env_vars(resolved):
        for cmd in plan.setup_commands:
            logger.info("[%s] setup: %s", group_id, cmd)
            subprocess.check_call(cmd, shell=True)

        for step in plan.steps:
            with _env_vars(step.env_vars):
                try:
                    for cmd in step.commands:
                        logger.info("[%s/%s] %s", group_id, step.test_id, cmd)
                        subprocess.check_call(cmd, shell=True)
                except subprocess.CalledProcessError:
                    hint = _rerun_hint(group_id, env_overrides, input_overrides)
                    logger.error(
                        "[%s/%s] failed. To rerun on RE:\n  %s",
                        group_id, step.test_id, hint,
                    )
                    raise
            logger.info("[%s/%s] passed", group_id, step.test_id)


# Flags consumed by RE submission, not passed to the remote command.
_RE_FLAGS = {"--re", "--pr", "--commit", "--dry-run", "--no-follow", "--interactive"}


def _build_remote_command() -> str:
    """Rebuild the user's command from sys.argv, stripping RE-only flags."""
    args = sys.argv[1:]  # drop 'lumen' (entry point)
    filtered = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in _RE_FLAGS:
            # --pr and --commit take a value; the rest are boolean
            if arg in ("--pr", "--commit", "--interactive") and i + 1 < len(args):
                skip_next = True
            continue
        filtered.append(arg)
    return "lumen " + " ".join(shlex.quote(a) for a in filtered)


class PytorchTestRunner:
    def __init__(self, args: Any) -> None:
        self.group_id: str = args.group_id
        raw_inputs = getattr(args, "input", []) or []
        self.input_overrides = dict(i.split("=", 1) for i in raw_inputs)
        raw_envs = getattr(args, "env", []) or []
        self.env_overrides = dict(e.split("=", 1) for e in raw_envs)
        self.re = getattr(args, "re", False)
        self.pr = getattr(args, "pr", None)
        self.commit = getattr(args, "commit", None)
        self.dry_run = getattr(args, "dry_run", False)
        self.no_follow = getattr(args, "no_follow", False)
        self.show_hint = getattr(args, "show_hint", False)
        self.interactive = getattr(args, "interactive", None)
        if self.interactive is not None and self.interactive > 240:
            raise ValueError("--interactive max is 240 minutes (4 hours)")

    def run(self) -> None:
        if self.group_id not in LINT_PLANS:
            raise RuntimeError(
                f"group '{self.group_id}' not found. Available: {sorted(LINT_PLANS)}"
            )
        plan = LINT_PLANS[self.group_id]

        if self.re:
            submit_command(
                command=_build_remote_command(),
                name=f"lint-{self.group_id}",
                pr=self.pr,
                commit=self.commit,
                dry_run=self.dry_run,
                no_follow=self.no_follow,
                image=plan.image,
                idle_timeout=self.interactive,
            )
        else:
            run_plan(self.group_id, plan, self.env_overrides, self.input_overrides)
            if self.show_hint:
                hint = _rerun_hint(self.group_id, self.env_overrides, self.input_overrides)
                logger.info("To rerun on RE:\n  %s", hint)
