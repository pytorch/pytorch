"""
Lint test runner — local execution.

Usage:
    lumen test lint --group-id lintrunner_noclang
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from collections.abc import Iterator
from typing import Any

from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS, TestPlan


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


def run_plan(group_id: str, plan: TestPlan, input_overrides: dict[str, str] | None = None) -> None:
    """Run a test plan locally."""
    logger.info("[%s] %s", group_id, plan.title)

    resolved = plan.resolve_env_vars(input_overrides)
    with _env_vars(resolved):
        for cmd in plan.setup_commands:
            logger.info("[%s] setup: %s", group_id, cmd)
            subprocess.check_call(cmd, shell=True)

        for step in plan.steps:
            with _env_vars(step.env_vars):
                for cmd in step.commands:
                    logger.info("[%s/%s] %s", group_id, step.test_id, cmd)
                    subprocess.check_call(cmd, shell=True)
            logger.info("[%s/%s] passed", group_id, step.test_id)


class PytorchTestRunner:
    def __init__(self, args: Any) -> None:
        self.group_id: str = args.group_id
        raw_inputs = getattr(args, "input", []) or []
        self.input_overrides = dict(i.split("=", 1) for i in raw_inputs)

    def run(self) -> None:
        if self.group_id not in LINT_PLANS:
            raise RuntimeError(
                f"group '{self.group_id}' not found. "
                f"Available: {sorted(LINT_PLANS)}"
            )
        plan = LINT_PLANS[self.group_id]
        run_plan(self.group_id, plan, self.input_overrides)
