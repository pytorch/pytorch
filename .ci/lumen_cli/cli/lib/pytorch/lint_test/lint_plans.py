"""
Lint test plans — maps lint-osdc.yml jobs to lumen test plans.

Usage:
    lumen test lint --group-id lintrunner_noclang
    lumen test lint --group-id lintrunner_noclang --remote-execution --pr 178213
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestStep:
    test_id: str
    commands: list[str]
    env_vars: dict[str, str] = field(default_factory=dict)


DEFAULT_BOOTSTRAP = ["git_clone", "setup_uv"]


@dataclass
class LintTestPlan:
    group_id: str
    title: str
    image: str
    steps: list[TestStep] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)
    bootstrap: list[str] = field(default_factory=lambda: list(DEFAULT_BOOTSTRAP))
    # Declarative inputs with defaults. CLI can override via --input key=value.
    inputs: dict[str, str] = field(default_factory=dict)

    def resolve_env_vars(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """Resolve env_vars by substituting {input_name} placeholders.

        Special handling: if an input value is "*", the placeholder
        {input_name} is replaced with "--all-files" (matching the
        CHANGED_FILES convention in lint-osdc.yml).
        """
        values = {**self.inputs, **(overrides or {})}
        resolved = {}
        for k, v in values.items():
            if v == "*":
                resolved[k] = "--all-files"
            else:
                resolved[k] = v
        return {k: v.format(**resolved) for k, v in self.env_vars.items()}


# Common setup shared by all _lint.yml based jobs
_LINT_SETUP = [
    "python -m pip install -r .ci/docker/requirements-ci.txt",
    "dnf install -y doxygen graphviz nodejs npm",
    "npm install -g markdown-toc",
    "lintrunner init",
]

LINT_PLANS: dict[str, LintTestPlan] = {
    "lintrunner_noclang": LintTestPlan(
        group_id="lintrunner_noclang",
        title="Lintrunner (no clang)",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        inputs={"changed_files": "*"},
        env_vars={
            "ADDITIONAL_LINTRUNNER_ARGS": "--skip CLANGTIDY,CLANGTIDY_EXECUTORCH_COMPATIBILITY,CLANGFORMAT,PYREFLY {changed_files}",
        },
        steps=[
            TestStep(
                test_id="noclang",
                commands=["bash .github/scripts/lintrunner.sh"],
            ),
        ],
    ),
    "lintrunner_clang": LintTestPlan(
        group_id="lintrunner_clang",
        title="Lintrunner (clang)",
        image="ghcr.io/pytorch/test-infra:cuda-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        inputs={"changed_files": "*"},
        env_vars={
            "ADDITIONAL_LINTRUNNER_ARGS": "--take CLANGTIDY,CLANGFORMAT {changed_files}",
            "CLANG": "1",
        },
        steps=[
            TestStep(
                test_id="clang",
                commands=["bash .github/scripts/lintrunner.sh"],
            ),
        ],
    ),
    "lintrunner_pyrefly": LintTestPlan(
        group_id="lintrunner_pyrefly",
        title="Lintrunner (pyrefly)",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        env_vars={
            "ADDITIONAL_LINTRUNNER_ARGS": "--take PYREFLY --all-files",
        },
        steps=[
            TestStep(
                test_id="pyrefly",
                commands=["bash .github/scripts/lintrunner.sh"],
            ),
        ],
    ),
    "quick_checks": LintTestPlan(
        group_id="quick_checks",
        title="Quick checks",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        steps=[
            TestStep(
                test_id="quick_checks",
                commands=[
                    # No non-breaking spaces
                    """(! git --no-pager grep -In "$(printf '\\xC2\\xA0')" -- . || (echo "Non-breaking spaces found"; false))""",
                    # Cross-OS compatible file names
                    """(! git ls-files | grep -E '([<>:"|?*]|[ .]$)' || (echo "Invalid file names found"; false))""",
                    # No versionless Python shebangs
                    """(! git --no-pager grep -In '#!.*python$' -- . || (echo "Versionless Python shebangs found"; false))""",
                    # Ciflow tags
                    "python3 .github/scripts/collect_ciflow_labels.py --validate-tags",
                    # C++ docs check
                    "pushd docs/cpp/source && ./check-doxygen.sh && popd",
                    # CUDA kernel launch check
                    "python3 torch/testing/_internal/check_kernel_launches.py",
                ],
            ),
        ],
    ),
    "workflow_checks": LintTestPlan(
        group_id="workflow_checks",
        title="Workflow checks",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        steps=[
            TestStep(
                test_id="workflow_checks",
                commands=[
                    ".github/scripts/generate_ci_workflows.py",
                    ".github/scripts/report_git_status.sh .github/workflows",
                    ".github/scripts/ensure_actions_will_cancel.py",
                ],
            ),
        ],
    ),
    "toc": LintTestPlan(
        group_id="toc",
        title="Table of contents check",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        steps=[
            TestStep(
                test_id="toc",
                commands=[
                    """export PATH=~/.npm-global/bin:"$PATH" && for FILE in $(git grep -Il '<!-- toc -->' -- '**.md'); do markdown-toc --bullets='-' -i "$FILE"; done""",
                    ".github/scripts/report_git_status.sh .",
                ],
            ),
        ],
    ),
    "test_tools": LintTestPlan(
        group_id="test_tools",
        title="Test tools",
        image="ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930",
        setup_commands=_LINT_SETUP,
        steps=[
            TestStep(
                test_id="test_tools",
                commands=[
                    "PYTHONPATH=$(pwd) pytest tools/stats",
                    'PYTHONPATH=$(pwd) pytest tools/test -o "python_files=test*.py"',
                    'PYTHONPATH=$(pwd) pytest .github/scripts -o "python_files=test*.py"',
                ],
            ),
        ],
    ),
}
