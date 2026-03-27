"""Remote execution support — submit commands to run on RE infrastructure."""

from __future__ import annotations

import logging
from pathlib import Path

from re_cli.core.core_types import StepConfig
from re_cli.core.job_runner import JobRunner
from re_cli.core.k8s_client import K8sClient, K8sConfig
from re_cli.core.script_builder import RunnerScriptBuilder

from cli.lib.pytorch.git_utils import CommitResolver


logger = logging.getLogger(__name__)

SCRIPT_MODULES_DIR = Path(__file__).resolve().parent / "script_modules"

DEFAULT_IMAGE = "ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930"
DEFAULT_BOOTSTRAP = ["git_clone", "setup_uv", "check_python", "install_lumen"]


def _load_script_module(name: str) -> str:
    """Load a bash script from script_modules/ by name (without .sh)."""
    path = SCRIPT_MODULES_DIR / f"{name}.sh"
    if not path.exists():
        raise RuntimeError(f"script module '{name}' not found at {path}")
    template = path.read_text()
    return "\n".join(
        line
        for line in template.splitlines()
        if not line.startswith("#") and not line.startswith("set -")
    ).strip()


class LumenScriptBuilder(RunnerScriptBuilder):
    """RE script builder for lumen."""

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

    def add_check_python(self) -> LumenScriptBuilder:
        return self._add_script("check_python", "check_python")

    def add_install_lumen(self) -> LumenScriptBuilder:
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: install_lumen\n# {'=' * 44}\n"
            "uv pip install -e .ci/lumen_cli"
        )
        return self


REPO = "https://github.com/pytorch/pytorch.git"


def submit_command(
    command: str,
    name: str = "re-run",
    pr: int | None = None,
    commit: str | None = None,
    dry_run: bool = False,
    no_follow: bool = False,
    image: str = DEFAULT_IMAGE,
    bootstrap: list[str] | None = None,
) -> None:
    """Submit a command to Remote Execution."""
    if bootstrap is None:
        bootstrap = list(DEFAULT_BOOTSTRAP)

    modules_list = (
        ["header", "find_script", "git_clone", "git_checkout"]
        + bootstrap
        + ["run_script", "upload_outputs"]
    )
    seen: set[str] = set()
    modules = [m for m in modules_list if not (m in seen or seen.add(m))]

    step = StepConfig(
        name=name,
        command=command,
        task_type="cpu-large",
        image=image,
        runner_modules=modules,
        env_vars={},
    )

    resolver = CommitResolver(REPO)
    resolved = resolver.resolve(pr, commit)

    client = K8sClient(K8sConfig(namespace="remote-execution-system", timeout=60))
    runner = JobRunner(
        client=client,
        name=name,
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


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def _run_re(args) -> None:
    submit_command(
        command=args.command,
        pr=getattr(args, "pr", None),
        commit=getattr(args, "commit", None),
        dry_run=getattr(args, "dry_run", False),
        no_follow=getattr(args, "no_follow", False),
    )


def register_re_commands(subparsers) -> None:
    """Register `lumen re` subcommand."""
    import argparse

    parser = subparsers.add_parser(
        "re",
        help="Submit a command to Remote Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--command",
        required=True,
        help="command to run remotely, e.g. 'echo hello'",
    )
    parser.add_argument("--pr", type=int, help="PR number (auto-detected if omitted)")
    parser.add_argument("--commit", help="commit SHA (skips PR detection)")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--no-follow", action="store_true", default=False)
    parser.set_defaults(func=_run_re)
