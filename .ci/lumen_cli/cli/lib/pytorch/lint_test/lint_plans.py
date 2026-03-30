"""
Lint test plans — maps lint-osdc.yml jobs to lumen test plans.

Usage:
    lumen test lint --group-id lintrunner_noclang
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestStep:
    test_id: str
    commands: list[str]
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class TestPlan:
    group_id: str
    title: str
    image: str
    steps: list[TestStep] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)
    # Declarative inputs with defaults. CLI can override via --input key=value.
    inputs: dict[str, str] = field(default_factory=dict)

    def resolve_env_vars(
        self, env_overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Return env_vars merged with CLI --env overrides."""
        return {**self.env_vars, **(env_overrides or {})}


# Common setup shared by all _lint.yml based jobs
_LINT_SETUP = [
    "python -m pip install -r .ci/docker/requirements-ci.txt",
    "dnf install -y doxygen graphviz nodejs npm",
    "npm install -g markdown-toc",
    "lintrunner init",
]

LINT_PLANS: dict[str, TestPlan] = {
    "lintrunner_noclang": TestPlan(
        group_id="lintrunner_noclang",
        title="Lintrunner (no clang)",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        env_vars={
            "ADDITIONAL_LINTRUNNER_ARGS": "--skip CLANGTIDY,CLANGTIDY_EXECUTORCH_COMPATIBILITY,CLANGFORMAT,PYREFLY --all-files",
        },
        steps=[
            TestStep(
                test_id="noclang",
                commands=[
                    "echo \"ADDITIONAL_LINTRUNNER_ARGS=$ADDITIONAL_LINTRUNNER_ARGS\"",
                    "bash .github/scripts/lintrunner.sh",
                ],
            ),
        ],
    ),
}
