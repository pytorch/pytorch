"""
Lint test runner — local execution or remote execution via RE.

Local:
    lumen test lint --group-id lintrunner_noclang

Remote execution:
    lumen test lint --group-id lintrunner_noclang --remote-execution --pr 178213
    lumen test lint --group-id lintrunner_noclang --remote-execution --pr 178213 --dry-run
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS, LintTestPlan


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def generate_script(plan: LintTestPlan, input_overrides: dict[str, str] | None = None) -> str:
    """Generate a self-contained bash script from a lint test plan."""
    parts = ["#!/bin/bash", "set -euo pipefail", ""]

    # env vars (resolved with inputs)
    env_vars = plan.resolve_env_vars(input_overrides)
    for k, v in env_vars.items():
        parts.append(f'export {k}="{v}"')
    for step in plan.steps:
        for k, v in step.env_vars.items():
            parts.append(f'export {k}="{v}"')
    if env_vars or any(s.env_vars for s in plan.steps):
        parts.append("")

    # setup
    if plan.setup_commands:
        parts.append("# === setup ===")
        parts.extend(plan.setup_commands)
        parts.append("")

    # steps
    for step in plan.steps:
        parts.append(f"# === {step.test_id} ===")
        parts.extend(step.commands)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def run_plan(group_id: str, plan: LintTestPlan, input_overrides: dict[str, str] | None = None) -> None:
    """Run a lint test plan locally."""
    logger.info("[%s] %s", group_id, plan.title)

    env_backup = {}
    resolved = plan.resolve_env_vars(input_overrides)
    for k, v in resolved.items():
        env_backup[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        for cmd in plan.setup_commands:
            logger.info("[%s] setup: %s", group_id, cmd)
            subprocess.check_call(cmd, shell=True)

        for step in plan.steps:
            for k, v in step.env_vars.items():
                env_backup.setdefault(k, os.environ.get(k))
                os.environ[k] = v
            for cmd in step.commands:
                logger.info("[%s/%s] %s", group_id, step.test_id, cmd)
                subprocess.check_call(cmd, shell=True)
            logger.info("[%s/%s] passed", group_id, step.test_id)
    finally:
        for k, orig in env_backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


class PytorchTestRunner(BaseRunner):
    def __init__(self, args: Any) -> None:
        self.group_id: str = args.group_id
        self.remote_execution: bool = getattr(args, "remote_execution", False)
        self.pr: int | None = getattr(args, "pr", None)
        self.commit: str | None = getattr(args, "commit", None)
        self.dry_run: bool = getattr(args, "dry_run", False)
        self.no_follow: bool = getattr(args, "no_follow", False)
        raw_inputs = getattr(args, "input", []) or []
        self.input_overrides = dict(i.split("=", 1) for i in raw_inputs)

    def run(self) -> None:
        if self.group_id not in LINT_PLANS:
            raise RuntimeError(
                f"group '{self.group_id}' not found. "
                f"Available: {sorted(LINT_PLANS)}"
            )
        plan = LINT_PLANS[self.group_id]

        if self.remote_execution:
            from cli.lib.pytorch.re_runner import submit_to_re
            submit_to_re(
                plan,
                pr=self.pr,
                commit=self.commit,
                dry_run=self.dry_run,
                no_follow=self.no_follow,
                input_overrides=self.input_overrides,
            )
            return

        run_plan(self.group_id, plan, self.input_overrides)
