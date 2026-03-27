"""
Lint test runner.

Usage:
    lumen test lint --group-id lintrunner_noclang
"""

from __future__ import annotations

import logging
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS, LintTestPlan


logger = logging.getLogger(__name__)


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


class PytorchTestRunner(BaseRunner):
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
        print(generate_script(plan, self.input_overrides))
