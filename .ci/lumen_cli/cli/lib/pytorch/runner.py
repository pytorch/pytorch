"""
Lint test runner.

Usage:
    lumen test lint --group-id lintrunner_noclang
"""

from __future__ import annotations

import logging
from typing import Any

from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS


logger = logging.getLogger(__name__)


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
        logger.info("Plan: %s", plan)
