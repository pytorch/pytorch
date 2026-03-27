"""Remote execution support for lint test plans."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from cli.lib.pytorch.lint_test.lint_plans import LintTestPlan
from re_cli.core.core_types import StepConfig
from re_cli.core.job_runner import JobRunner
from re_cli.core.k8s_client import K8sClient, K8sConfig
from re_cli.core.script_builder import RunnerScriptBuilder


logger = logging.getLogger(__name__)

REPO = "https://github.com/pytorch/pytorch.git"
SCRIPT_MODULES_DIR = Path(__file__).resolve().parent / "script_modules"


def _load_script_module(name: str) -> str:
    """Load a bash script from script_modules/ by name (without .sh)."""
    path = SCRIPT_MODULES_DIR / f"{name}.sh"
    if not path.exists():
        raise RuntimeError(f"script module '{name}' not found at {path}")
    template = path.read_text()
    return "\n".join(
        line for line in template.splitlines()
        if not line.startswith("#") and not line.startswith("set -")
    ).strip()


def _pr_info(pr: int) -> dict:
    """Get commit SHA and repo URL from a PR number."""
    out = subprocess.run(
        [
            "gh", "pr", "view", str(pr),
            "--repo", "pytorch/pytorch",
            "--json", "headRefOid,headRefName,headRepository,headRepositoryOwner",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    owner = data["headRepositoryOwner"]["login"]
    repo_name = data["headRepository"]["name"]
    return {
        "sha": data["headRefOid"],
        "branch": data["headRefName"],
        "repo": f"https://github.com/{owner}/{repo_name}.git",
    }


def _resolve_commit(pr: int | None, commit: str | None) -> dict:
    if pr:
        info = _pr_info(pr)
        logger.info("PR #%d -> %s (%s)", pr, info["sha"][:12], info["repo"])
        return {"sha": info["sha"], "repo": info["repo"]}
    if commit:
        return {"sha": commit, "repo": REPO}
    raise RuntimeError("--pr or --commit is required for remote execution")


def _build_run_command(plan: LintTestPlan, input_overrides: dict[str, str] | None = None) -> str:
    """Build the lumen CLI command that RE will execute after bootstrap."""
    cmd = f"lumen test lint --group-id {plan.group_id}"
    if input_overrides:
        for k, v in input_overrides.items():
            cmd += f" --input {k}='{v}'"
    return cmd


class LumenScriptBuilder(RunnerScriptBuilder):
    """RE script builder for lumen lint plans."""

    DEFAULT_MODULES = [
        "header",
        "find_script",
        "git_clone",
        "git_checkout",
        "run_script",
        "upload_outputs",
    ]

    def _add_script(self, module_name: str, script_name: str) -> LumenScriptBuilder:
        body = _load_script_module(script_name)
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: {module_name}\n# {'=' * 44}\n" + body
        )
        return self

    def add_git_clone(self) -> LumenScriptBuilder:
        return self._add_script("git_clone", "git_clone")

    def add_setup_uv(self) -> LumenScriptBuilder:
        return self._add_script("setup_uv", "setup_uv")

    def add_install_lumen(self) -> LumenScriptBuilder:
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: install_lumen\n# {'=' * 44}\n"
            "(cd .ci/lumen_cli && python -m pip install -e .)"
        )
        return self


def submit_to_re(
    plan: LintTestPlan,
    pr: int | None,
    commit: str | None,
    dry_run: bool,
    no_follow: bool = False,
    input_overrides: dict[str, str] | None = None,
) -> None:
    """Submit a lint test plan to Remote Execution."""
    command = _build_run_command(plan, input_overrides)

    # hard code for now for task_type for demoing
    step = StepConfig(
        name=plan.group_id,
        command=command,
        task_type="cpu-large",
        image=plan.image,
    )
    job_name = f"{plan.group_id}-pr{pr}" if pr else plan.group_id

    resolved = _resolve_commit(pr, commit)
    client = K8sClient(K8sConfig(namespace="remote-execution-system", timeout=60))
    runner = JobRunner(
        client=client,
        name=job_name,
        step_configs=[step],
        script_builder_class=LumenScriptBuilder,
    )
    runner.run(
        commit=resolved["sha"],
        repo=resolved["repo"],
        follow=not no_follow,
        dry_run=dry_run,
    )
    if runner.run_id:
        print(f"\nRun ID: {runner.run_id}")
        print(f"Stream:  blast stream {runner.run_id}")
