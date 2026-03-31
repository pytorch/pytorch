from __future__ import annotations

import json
import logging
import os
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory
from cli.lib.pytorch.base import (
    BasePytorchTestPlan,
    matches_env,
    resolve_env_vars,
    TestStep,
)
from cli.lib.pytorch.plans import PYTORCH_TEST_LIBRARY


logger = logging.getLogger(__name__)


def resolve_plans_for_env(
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> list[str]:
    """Return group_ids eligible for the given build_env (ignores TEST_CONFIG)."""
    registry = library if library is not None else PYTORCH_TEST_LIBRARY
    return [gid for gid, plan in registry.items() if plan.is_eligible(build_env)]


def resolve_plans_for_test_config(
    test_config: str,
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> list[str]:
    """
    Return all group_ids matching TEST_CONFIG + build_env, in registry order.
    Multiple plans can match and will be run in sequence.
    Raises RuntimeError if no plan matches.
    """
    if not build_env:
        raise RuntimeError("build_env is required and must be non-empty")
    registry = library if library is not None else PYTORCH_TEST_LIBRARY
    matched = [
        gid
        for gid, plan in registry.items()
        if plan.is_eligible(build_env, test_config)
    ]
    if not matched:
        raise RuntimeError(
            f"No plan matched TEST_CONFIG={test_config!r} build_env={build_env!r}. "
            f"Available group_ids: {sorted(registry)}"
        )
    return matched


# ---------------------------------------------------------------------------
# Repro context + runner
# ---------------------------------------------------------------------------


def _build_repro_context(
    plan: BasePytorchTestPlan,
    step: TestStep,
    build_env: str,
) -> dict:
    return {
        "plan_env_vars": resolve_env_vars(plan.env_vars, build_env),
        "step_env_vars": resolve_env_vars(step.env_vars, build_env),
        "pip_installs": plan.pip_installs + step.pip_installs,
        "working_dir": step.working_dir or plan.working_dir,
    }


def _print_repro(
    group_id: str,
    step: TestStep,
    build_env: str,
    ctx: dict,
    cmd: str | None = None,
) -> None:
    lumen_cmd = (
        f"lumen test pytorch-core"
        f" --group-id {group_id}"
        f" --build-env {build_env}"
        f" --test-id {step.test_id}"
        f" --no-upload"
    )
    for k, v in step.params.items():
        lumen_cmd += f" --filter {k}={v}"
    if cmd:
        lumen_cmd += f' --cmd "{cmd}"'

    manual: list[str] = []
    for pip_args in ctx["pip_installs"]:
        manual.append(f"  pip install {' '.join(pip_args)}")
    if ctx["working_dir"]:
        manual.append(f"  cd {ctx['working_dir']}")
    for k, v in {**ctx["plan_env_vars"], **ctx["step_env_vars"]}.items():
        manual.append(f"  export {k}={v}")
    if cmd:
        manual.append(f"  {cmd}")
    else:
        manual.append("  # then run your command directly")

    lines = [
        f"\nTo reproduce {group_id}/{step.test_id}:",
        f"  {lumen_cmd}",
    ]
    logger.error("\n".join(lines))


def run_test_plan(
    group_id: str,
    build_env: str,
    test_id: str | None = None,
    cmd: str | None = None,
    filters: dict[str, str] | None = None,
    shard_id: int = 1,
    num_shards: int = 1,
    no_upload: bool = False,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> None:
    """
    Run a test plan (or a subset of its steps) from the registry.

    Args:
        group_id:   Key in PYTORCH_TEST_LIBRARY, e.g. "pytorch_jit_legacy".
        test_id:    If set, run only the step with this exact id.
        filters:    Key=value pairs matched against step.params — useful for
                    reproducing a specific combo, e.g. {"mode": "training"}.
                    All filters must match (AND logic). Combined with test_id
                    when both are provided.
        cmd:        If set, replay the setup context of test_id but run this
                    command instead of step.fn(). Requires test_id to be set.
                    Use this to reproduce a specific pytest line within a step.
        shard_id:   Current shard (1-based).
        num_shards: Total number of shards.
        no_upload:  If True, sets LUMEN_NO_UPLOAD=1 so upload flags are stripped
                    from run_test.py invocations (useful for local reproduction).
        library:    Override the default registry (useful for testing).
    """
    if not build_env:
        raise RuntimeError("build_env is required and must be non-empty")
    if cmd and not test_id:
        raise RuntimeError("--cmd requires --test-id to identify the setup context")
    registry = library if library is not None else PYTORCH_TEST_LIBRARY

    if group_id not in registry:
        raise RuntimeError(
            f"group '{group_id}' not found. Available: {sorted(registry)}"
        )
    plan = registry[group_id]
    if plan.run_on and not any(matches_env(c, build_env) for c in plan.run_on):
        logger.warning(
            "[%s] run_on conditions %s do not match build_env=%r — running anyway",
            group_id,
            plan.run_on,
            build_env,
        )
    all_steps = plan.get_steps(build_env, shard_id=shard_id, num_shards=num_shards)

    steps = all_steps
    if test_id:
        steps = [s for s in steps if s.test_id == test_id]
    if filters:
        steps = [
            s for s in steps if all(s.params.get(k) == v for k, v in filters.items())
        ]

    if not steps:
        raise RuntimeError(
            f"No steps matched in '{group_id}' "
            f"(test_id={test_id!r}, filters={filters}). "
            f"Available: {[s.test_id for s in all_steps]}"
        )

    if plan.setup_fn:
        logger.info("[%s] running setup", group_id)
        _env_before = dict(os.environ)
        plan.setup_fn()
        _env_changes = {k: v for k, v in os.environ.items() if _env_before.get(k) != v}
        if _env_changes:
            logger.warning(
                "[%s] setup_fn set env vars outside temp_environ context (not auto-cleaned): %s",
                group_id,
                list(_env_changes.keys()),
            )

    failures: list[str] = []

    upload_env = {"LUMEN_NO_UPLOAD": "1"} if no_upload else {}

    first_ctx = _build_repro_context(plan, steps[0], build_env)
    with temp_environ({**first_ctx["plan_env_vars"], **upload_env}):
        for step in steps:
            ctx = _build_repro_context(plan, step, build_env)
            logger.info("[%s/%s] starting", group_id, step.test_id)

            for pip_args in ctx["pip_installs"]:
                pip_install_packages(pip_args)

            lumen_msg = (
                f"Or via lumen: "
                f"lumen test pytorch-core"
                f" --group-id {group_id}"
                f" --build-env {build_env}"
                f" --test-id {step.test_id}"
                f" --no-upload"
                f' --cmd "{{repro}}"'
            )
            with (
                temp_environ(
                    {
                        **ctx["step_env_vars"],
                        "PYTORCH_EXTRA_REPRO_MESSAGE": lumen_msg,
                    }
                ),
                working_directory(ctx["working_dir"] or ""),
            ):
                try:
                    if step.setup_fn:
                        logger.info(
                            "[%s/%s] running step setup", group_id, step.test_id
                        )
                        step.setup_fn()
                    if cmd:
                        logger.info(
                            "[%s/%s] running custom cmd: %s",
                            group_id,
                            step.test_id,
                            cmd,
                        )
                        run_command(cmd, use_shell=True)
                    else:
                        step.fn()
                    logger.info("[%s/%s] passed", group_id, step.test_id)
                except Exception as e:
                    logger.error("[%s/%s] FAILED: %s", group_id, step.test_id, str(e))  # noqa: G200
                    _print_repro(group_id, step, build_env, ctx, cmd=cmd)
                    failures.append(step.test_id)

    if failures:
        raise RuntimeError(f"[{group_id}] {len(failures)} step(s) failed: {failures}")


# ---------------------------------------------------------------------------
# Workflow dry-run
# ---------------------------------------------------------------------------


def dry_run_from_workflow(
    yaml_path: str,
    runner_filter: str = "linux.2xlarge",
) -> None:
    """Parse a GitHub Actions workflow YAML and print lumen commands (dry-run)."""
    from cli.lib.pytorch.workflow_parser import parse_workflow

    entries = parse_workflow(yaml_path, runner_filter)

    if not entries:
        print(f"No entries found matching runner_filter={runner_filter!r}")
        return

    # Group by build_env
    by_build_env: dict[str, list[dict]] = {}
    for entry in entries:
        by_build_env.setdefault(entry["build_env"], []).append(entry)

    for build_env, group in by_build_env.items():
        build_runner = group[0]["build_runner"]
        print(f"{build_env}  (build: {build_runner})")
        for entry in group:
            config = entry["config"]
            test_runner = entry["test_runner"]
            lumen_cmd = (
                f"lumen test pytorch-core"
                f" --test-config {config}"
                f" --build-env {build_env}"
                f" --no-upload"
            )
            print(f"  {config:<30} {test_runner}")
            print(f"    {lumen_cmd}")
        print()


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


class PytorchTestRunner(BaseRunner):
    def __init__(self, args: Any) -> None:
        self.group_id = getattr(args, "group_id", None)
        self.test_config = getattr(args, "test_config", None)
        self.build_env = getattr(args, "build_env", None)
        self.test_id = getattr(args, "test_id", None)
        self.cmd = getattr(args, "cmd", None)
        self.shard_id = getattr(args, "shard_id", None) or int(
            os.environ.get("SHARD_NUMBER", 1)
        )
        self.num_shards = getattr(args, "num_shards", None) or int(
            os.environ.get("NUM_TEST_SHARDS", 1)
        )
        self.no_upload = getattr(args, "no_upload", False)
        self.print_plan = getattr(args, "print_plan", False)
        self.from_workflow = getattr(args, "workflow", None)
        self.runner_filter = getattr(args, "runner_filter", "linux.2xlarge")
        raw_filters = getattr(args, "filter", []) or []
        self.filters = (
            dict(f.split("=", 1) for f in raw_filters) if raw_filters else None
        )

    def _print_plan(self, group_ids: list[str], registry: dict) -> None:
        print(f"build_env: {self.build_env}")
        print(f"plans ({len(group_ids)}):")
        for gid in group_ids:
            plan = registry[gid]
            steps = plan.get_steps(self.build_env, self.shard_id, self.num_shards)
            print(f"  {gid}  —  {plan.title}")
            for step in steps:
                print(f"    • {step.test_id}")

    def run(self) -> None:
        if self.from_workflow:
            from cli.lib.pytorch.workflow_parser import (
                parse_workflow,
                resolve_workflow_path,
            )

            path = resolve_workflow_path(self.from_workflow)
            if not self.test_config and not self.group_id:
                raise RuntimeError("--workflow requires --test-config or --group-id")
            entries = parse_workflow(str(path), runner_filter="")
            # filter by test_config
            config_filter = self.test_config
            if self.group_id:
                registry = PYTORCH_TEST_LIBRARY
                if self.group_id not in registry:
                    raise RuntimeError(f"group '{self.group_id}' not found")
                plan = registry[self.group_id]
                # use plan's test_configs as filter (string conditions only)
                config_filter = next(
                    (c for c in plan.test_configs if isinstance(c, str)), None
                )
            matched = [
                e for e in entries if config_filter and config_filter in e["config"]
            ]
            if not matched:
                print(f"No combos found for config={config_filter!r} in {path}")
                return
            output = []
            for e in matched:
                lumen_cmd = (
                    f"lumen test pytorch-core"
                    f" --test-config {e['config']}"
                    f" --build-env {e['build_env']}"
                )
                output.append(
                    {
                        "build_env": e["build_env"],
                        "build_runner": e["build_runner"],
                        "build_image": e.get("build_image", ""),
                        "config": e["config"],
                        "test_runner": e["test_runner"],
                        "lumen_cmd": lumen_cmd,
                    }
                )
            print(json.dumps(output, indent=2))
            return
        if not self.build_env:
            raise RuntimeError("--build-env is required")
        if not self.group_id and not self.test_config:
            raise RuntimeError("--group-id or --test-config is required")
        if self.group_id:
            # Explicit group_id: run exactly this plan (hardware mismatch → warning).
            group_ids = [self.group_id]
        else:
            # test_config path: find all plans matching test_config + build_env,
            # run them in sequence.
            group_ids = resolve_plans_for_test_config(
                test_config=self.test_config,
                build_env=self.build_env,
            )
            logger.info(
                "test_config=%r build_env=%r → %d plan(s) to run: %s",
                self.test_config,
                self.build_env,
                len(group_ids),
                group_ids,
            )
        registry = PYTORCH_TEST_LIBRARY
        if self.print_plan:
            self._print_plan(group_ids, registry)
            return
        for group_id in group_ids:
            run_test_plan(
                group_id=group_id,
                build_env=self.build_env,
                test_id=self.test_id,
                cmd=self.cmd,
                filters=self.filters,
                shard_id=self.shard_id,
                num_shards=self.num_shards,
                no_upload=self.no_upload,
            )
