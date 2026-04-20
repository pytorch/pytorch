from __future__ import annotations

import logging

from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory
from cli.lib.core.pytorch.plans.benchmark_tests import BENCHMARK_TEST_PLANS
from cli.lib.core.pytorch.plans.core_tests import CORE_TEST_PLANS
from cli.lib.core.pytorch.pytorch_test_library import (
    BasePytorchTestPlan,
    resolve_env_vars,
    TestStep,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Central registry — add new plan files here as they are created
# ---------------------------------------------------------------------------

PYTORCH_TEST_LIBRARY: dict[str, BasePytorchTestPlan] = {
    **CORE_TEST_PLANS,
    **BENCHMARK_TEST_PLANS,
}


def resolve_plans_for_env(
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> list[str]:
    """Return group_ids eligible for the given build_env (ignores TEST_CONFIG)."""
    registry = library if library is not None else PYTORCH_TEST_LIBRARY
    return [gid for gid, plan in registry.items() if plan.is_eligible(build_env)]


def resolve_plan_for_test_config(
    test_config: str,
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> str:
    """
    Resolve exactly one group_id from TEST_CONFIG + build_env.

    Replaces the if/elif dispatch at the bottom of test.sh. In test.sh:

        (cd .ci/lumen_cli && python -m pip install -e .)
        python -m cli.run test pytorch-core \\
            --test-config "$TEST_CONFIG" \\
            --build-env  "$BUILD_ENVIRONMENT" \\
            --shard-id   "$SHARD_NUMBER" \\
            --num-shards "$NUM_TEST_SHARDS"

    Raises RuntimeError if zero or more than one plan matches — ambiguity is
    a configuration error that should be fixed in the plan definitions.
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
    if len(matched) > 1:
        raise RuntimeError(
            f"Ambiguous match for TEST_CONFIG={test_config!r} build_env={build_env!r}: "
            f"{matched}. Tighten test_configs/run_on conditions."
        )
    return matched[0]


# ---------------------------------------------------------------------------
# Repro context + runner
# ---------------------------------------------------------------------------


def _build_repro_context(
    plan: BasePytorchTestPlan,
    step: TestStep,
    build_env: str,
) -> dict:
    """
    Build the reproducible context for a step.

    env_vars are kept separate (not merged) so the runner can apply them at
    the right scope:
      - plan_env_vars: set once around all steps, persist for the whole plan
      - step_env_vars: scoped to this step only, cleaned up after it finishes
    """
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

    manual: list[str] = []
    for pip_args in ctx["pip_installs"]:
        manual.append(f"  pip install {' '.join(pip_args)}")
    if ctx["working_dir"]:
        manual.append(f"  cd {ctx['working_dir']}")
    for k, v in {**ctx["plan_env_vars"], **ctx["step_env_vars"]}.items():
        manual.append(f"  export {k}={v}")
    manual.append(f"  # then run your command directly")

    lines = [
        f"\nTo reproduce {group_id}/{step.test_id}:",
        f"  # Option 1 — lumen (replays full setup automatically):",
        f"  {lumen_cmd}",
        f"  # Option 2 — manual:",
        *manual,
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
        group_id:   Key in PYTORCH_TEST_LIBRARY, e.g. "pytorch_cpuonly".
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
    all_steps = plan.get_steps(build_env)

    steps = all_steps
    if test_id:
        steps = [s for s in steps if s.test_id == test_id]
    if filters:
        steps = [s for s in steps if all(s.params.get(k) == v for k, v in filters.items())]

    if not steps:
        raise RuntimeError(
            f"No steps matched in '{group_id}' "
            f"(test_id={test_id!r}, filters={filters}). "
            f"Available: {[s.test_id for s in all_steps]}"
        )

    if plan.setup_fn:
        logger.info("[%s] running setup", group_id)
        plan.setup_fn()

    failures: list[str] = []

    upload_env = {"LUMEN_NO_UPLOAD": "1"} if no_upload else {}

    # Plan-level env_vars wrap all steps — set once, persist across the whole plan.
    first_ctx = _build_repro_context(plan, steps[0], build_env)
    with temp_environ({**first_ctx["plan_env_vars"], **upload_env}):
        for step in steps:
            ctx = _build_repro_context(plan, step, build_env)
            logger.info("[%s/%s] starting", group_id, step.test_id)

            for pip_args in ctx["pip_installs"]:
                pip_install_packages(pip_args)

            # Step-level env_vars are scoped to this step only.
            with (
                temp_environ(ctx["step_env_vars"]),
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
                    logger.error("[%s/%s] FAILED: %s", group_id, step.test_id, e)
                    _print_repro(group_id, step, build_env, ctx)
                    failures.append(step.test_id)

    if failures:
        raise RuntimeError(f"[{group_id}] {len(failures)} step(s) failed: {failures}")
