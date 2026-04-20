from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union


# ---------------------------------------------------------------------------
# Environment condition types
# ---------------------------------------------------------------------------

# A condition is either:
#   - a plain string tag  → matched as substring of BUILD_ENVIRONMENT
#     e.g. "cuda" matches "pytorch-linux-focal-cuda12.1-py3.10-gcc9-sm86"
#   - a callable          → receives BUILD_ENVIRONMENT string, returns bool
EnvCondition = Union[str, Callable[[str], bool]]

# env_vars can be a static dict or a callable that receives BUILD_ENVIRONMENT
# and returns the dict to apply. Use a callable when different environments
# need different values for the same plan.
#   static:   {"CUDA_VISIBLE_DEVICES": "0"}
#   dynamic:  lambda env: {"HIP_VISIBLE_DEVICES": "0,1,2,3"} if "rocm" in env
#                         else {"CUDA_VISIBLE_DEVICES": "0"}
EnvVarsSpec = Union[dict[str, str], Callable[[str], dict[str, str]]]


def matches_env(condition: EnvCondition, build_env: str) -> bool:
    if callable(condition):
        return condition(build_env)
    return condition in build_env


def resolve_env_vars(spec: EnvVarsSpec, build_env: str) -> dict[str, str]:
    return spec(build_env) if callable(spec) else spec


# ---------------------------------------------------------------------------
# EnvMap — declarative env-conditional value resolution
#
# Instead of a lambda, use a dict keyed by EnvCondition.
# First matching condition wins.
#
#   device={is_xpu: "xpu", "cpu": "cpu", is_cuda: "cuda", is_rocm: "cuda"}
#   modes={is_rocm: ["training", "inference"], is_cuda: ["training"]}
#   dtype={is_rocm: "amp", is_cuda: ["float16", "amp"]}
#
# Also accepts a plain value (str / list) — passed through unchanged.
# ---------------------------------------------------------------------------

# EnvMap: dict keyed by EnvCondition, first matching key wins.
# Values can be any type — str, list[str], etc.
EnvMap = dict[EnvCondition, Any]


def resolve(spec: Any, build_env: str) -> Any:
    """
    Resolve an EnvMap or plain value against build_env.

    - Plain value (str, list, …) → returned as-is.
    - dict (EnvMap)              → first key (EnvCondition) that matches wins.
                                   Raises RuntimeError if nothing matches.
    """
    if not isinstance(spec, dict):
        return spec
    for condition, value in spec.items():
        if matches_env(condition, build_env):
            return value
    raise RuntimeError(
        f"No condition in EnvMap matched build_env={build_env!r}. "
        f"Conditions: {list(spec.keys())}"
    )


def resolve_to_list(spec: Any, build_env: str) -> list[str]:
    """Resolve spec and normalise to list[str] (wraps a bare str in a list)."""
    value = resolve(spec, build_env)
    return [value] if isinstance(value, str) else list(value)


# Convenience type aliases for BenchmarkTestPlan fields.
# Each accepts a plain value or an EnvMap keyed by EnvCondition.
DeviceSpec = Union[str, EnvMap]             # resolves to str
ModeSpec   = Union[list[str], EnvMap]       # resolves to list[str]
DtypeSpec  = Union[str, list[str], EnvMap]  # resolves to str | list[str] → normalised to list[str]


# ---------------------------------------------------------------------------
# Common pre-built conditions for readability in plan definitions
# ---------------------------------------------------------------------------

def is_cuda(env: str) -> bool:
    return "cuda" in env and "rocm" not in env

def is_rocm(env: str) -> bool:
    return "rocm" in env

def is_xpu(env: str) -> bool:
    return "xpu" in env

def is_gpu(env: str) -> bool:
    return is_cuda(env) or is_rocm(env) or is_xpu(env)

def is_cpu_only(env: str) -> bool:
    return not is_gpu(env)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TestStep:
    test_id: str
    fn: Callable[[], None]
    # Arbitrary key=value metadata describing this step's combo (model, mode, dtype…).
    # Used by run_test_plan --filter to select a subset of steps at repro time.
    params: dict[str, str] = field(default_factory=dict)
    env_vars: EnvVarsSpec = field(default_factory=dict)
    # Each inner list is a separate pip install invocation, preserving flags and order.
    # e.g. [["--pre", "torchao", "--index-url", "https://..."], ["-e", "."]]
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
    # Step-level setup — runs before this step's fn(), for both normal runs and repro.
    # Executes after plan-level setup_fn, pip_installs, and env_vars are applied.
    setup_fn: Callable[[], None] | None = None


@dataclass
class BasePytorchTestPlan:
    """
    Shared foundation for all PyTorch test plan types.

    run_on:
        Which BUILD_ENVIRONMENT values trigger this plan. Each entry is a
        string tag (substring match) or a callable(build_env) -> bool.
        Empty = eligible on all environments.

    test_configs:
        Which TEST_CONFIG values trigger this plan. Same string/callable
        convention as run_on. Empty = matches any TEST_CONFIG.

    Both conditions must pass for a plan to be selected. This mirrors the
    if/elif dispatch in test.sh where both TEST_CONFIG and BUILD_ENVIRONMENT
    are checked to route to the right test function.

    env_vars:
        Static dict or callable(build_env) -> dict. Step-level wins on conflict.
    """

    group_id: str
    title: str
    steps: list[TestStep]
    env_vars: EnvVarsSpec = field(default_factory=dict)
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
    # Always runs before any step, whether executing the full plan or reproducing
    # a single step. If setup is required for a test to run, it belongs here.
    setup_fn: Callable[[], None] | None = None
    # BUILD_ENVIRONMENT filter — empty = all environments
    run_on: list[EnvCondition] = field(default_factory=list)
    # TEST_CONFIG filter — empty = all configs
    test_configs: list[EnvCondition] = field(default_factory=list)

    def is_eligible(self, build_env: str, test_config: str = "") -> bool:
        env_ok = not self.run_on or any(
            matches_env(c, build_env) for c in self.run_on
        )
        config_ok = not self.test_configs or any(
            matches_env(c, test_config) for c in self.test_configs
        )
        return env_ok and config_ok

    def get_steps(self, build_env: str) -> list[TestStep]:
        """
        Return the steps to run for the given build_env.
        Override in subclasses for custom step generation.
        """
        return self.steps


# ---------------------------------------------------------------------------
# Concrete plan types
# ---------------------------------------------------------------------------


@dataclass
class CoreTestPlan(BasePytorchTestPlan):
    """Standard python test/run_test.py based tests."""
    pass


@dataclass
class BenchmarkTestPlan(BasePytorchTestPlan):
    """
    Inductor / dynamo benchmark tests.

    Steps are generated lazily via get_steps(build_env), which resolves
    env-conditional axes (device, modes, dtype) and expands the cartesian
    product of (models × modes × dtypes) into individual TestSteps.

    Each step carries a params dict so --filter can select any subset:
        --filter model=BERT_pytorch
        --filter mode=training
        --filter model=BERT_pytorch --filter dtype=float16

    device / modes / dtype accept:
        - a plain value  ("cuda", ["training"], "float16")
        - a list         (["float16", "amp"])
        - an EnvMap dict ({is_cuda: "cuda", is_rocm: "cuda", is_xpu: "xpu"})
    """

    device: DeviceSpec = "cuda"
    backend: str = "inductor"
    modes: ModeSpec = field(default_factory=list)
    dtype: DtypeSpec = "amp"
    suite: str = "torchbench"  # torchbench | huggingface | timm_models
    models: list[str] = field(default_factory=list)
    output_dir: str = "test/test-reports"
    extra_benchmark_flags: list[str] = field(default_factory=list)
    # Called once after all per-combo CSVs exist; receives per-combo output paths.
    join_results_fn: Callable[[list[str]], None] | None = None

    def output_path(self, model: str, mode: str, dtype: str, device: str) -> str:
        """
        Return the CSV output path for a single (model, mode, dtype, device) combo.

        Override to customise the output file naming convention.
        Called by get_steps() and join_results(); both use the same path so
        the join step can locate every per-combo file.
        """
        return os.path.join(
            self.output_dir,
            f"{self.backend}_{self.suite}_{model}_{mode}_{dtype}_{device}.csv",
        )

    def run_one(self, model: str, mode: str, dtype: str, device: str) -> None:
        """
        Run the benchmark script for a single (model, mode, dtype, device) combo.

        Override to change the invocation — e.g. different script, extra flags,
        or a completely different runner. All args are already resolved from the
        plan's EnvMap fields before this is called.
        """
        from cli.lib.core.pytorch.run_test_helper import run_command_checked

        flags = [
            f"--device {device}",
            f"--backend {self.backend}",
            f"--only {model}",
            f"--output {self.output_path(model, mode, dtype, device)}",
        ] + self.extra_benchmark_flags

        run_command_checked(
            f"python benchmarks/dynamo/{self.suite}.py "
            + " ".join(flags)
            + f" --{mode} --{dtype}"
        )

    def join_results(self, output_paths: list[str]) -> None:
        """
        Called once after all per-combo steps finish, with the list of CSV paths.

        Delegates to join_results_fn if provided; no-op otherwise.
        Override for custom post-processing (e.g. upload, comparison, alerting).
        """
        if self.join_results_fn is None:
            return
        self.join_results_fn(output_paths)

    def get_steps(self, build_env: str) -> list[TestStep]:
        """
        Resolve env-conditional axes and expand into one TestStep per
        (model × mode × dtype) combo, plus a final join_results step.

        Each step's params dict carries {"model", "mode", "dtype"} so that
        --filter can reproduce any subset without re-running the whole plan:
            --test-id BERT_pytorch                          # all combos for that model
            --test-id BERT_pytorch --filter mode=training  # specific combo

        Override entirely for custom step generation (e.g. subclass XYBenchmarkPlan).
        To change only the per-combo invocation, override run_one instead.
        To change only the output path format, override output_path instead.
        """
        device = resolve(self.device, build_env)
        modes  = resolve_to_list(self.modes, build_env)
        dtypes = resolve_to_list(self.dtype, build_env)

        output_paths = []
        steps = []
        for model in self.models:
            for mode in modes:
                for dtype in dtypes:
                    path = self.output_path(model, mode, dtype, device)
                    output_paths.append(path)
                    steps.append(TestStep(
                        test_id=model,
                        fn=functools.partial(self.run_one, model, mode, dtype, device),
                        params={"model": model, "mode": mode, "dtype": dtype},
                    ))

        steps.append(TestStep(
            test_id="join_results",
            fn=functools.partial(self.join_results, output_paths),
        ))
        return steps
