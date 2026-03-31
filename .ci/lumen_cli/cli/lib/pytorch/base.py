from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union

from cli.lib.common.utils import run_command


# ---------------------------------------------------------------------------
# Environment condition types
# ---------------------------------------------------------------------------

EnvCondition = Union[str, Callable[[str], bool]]  # noqa: UP007
EnvVarsSpec = Union[dict[str, str], Callable[[str], dict[str, str]]]  # noqa: UP007
ExtraArg = Union[str, Callable[[str], "str | None"]]  # noqa: UP007


def matches_env(condition: EnvCondition, build_env: str) -> bool:
    if callable(condition):
        return condition(build_env)
    return condition in build_env


def resolve_env_vars(spec: EnvVarsSpec, build_env: str) -> dict[str, str]:
    return spec(build_env) if callable(spec) else spec


def resolve_extra_arg_list(extra_args: list[ExtraArg], build_env: str) -> list[str]:
    result = []
    for arg in extra_args:
        val = arg(build_env) if callable(arg) else arg
        if val:
            result.append(val)
    return result


# ---------------------------------------------------------------------------
# EnvMap — declarative env-conditional value resolution
# ---------------------------------------------------------------------------

EnvMap = dict[EnvCondition, Any]


def resolve(spec: Any, build_env: str) -> Any:
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
    value = resolve(spec, build_env)
    return [value] if isinstance(value, str) else list(value)


DeviceSpec = Union[str, EnvMap]  # noqa: UP007
ModeSpec = Union[list[str], EnvMap]  # noqa: UP007
DtypeSpec = Union[str, list[str], EnvMap]  # noqa: UP007


# ---------------------------------------------------------------------------
# Common pre-built conditions
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
# run_test helpers
# ---------------------------------------------------------------------------

_UPLOAD_FLAGS = {"--upload-artifacts-while-running"}


def run_test(*args: str) -> None:
    """Invoke python test/run_test.py with the given arguments."""
    if os.environ.get("LUMEN_NO_UPLOAD"):
        args = tuple(a for a in args if a not in _UPLOAD_FLAGS)
    cmd = f"{sys.executable} test/run_test.py " + " ".join(args)
    if os.environ.get("LUMEN_DRY_RUN"):
        print(f"[dry-run] {cmd}")
        return
    run_command(cmd)


def run_command_checked(cmd: str) -> None:
    """Run an arbitrary shell command, raising on failure."""
    run_command(cmd, use_shell=True)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TestStep:
    test_id: str
    fn: Callable[[], None]
    params: dict[str, str] = field(default_factory=dict)
    env_vars: EnvVarsSpec = field(default_factory=dict)
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
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
    steps: list[TestStep] = field(default_factory=list)
    env_vars: EnvVarsSpec = field(default_factory=dict)
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
    setup_fn: Callable[[], None] | None = None
    run_on: list[EnvCondition] = field(default_factory=list)
    test_configs: list[EnvCondition] = field(default_factory=list)
    extra_args: list[ExtraArg] = field(default_factory=list)

    def is_eligible(self, build_env: str, test_config: str = "") -> bool:
        env_ok = not self.run_on or any(matches_env(c, build_env) for c in self.run_on)
        config_ok = not self.test_configs or any(
            matches_env(c, test_config) for c in self.test_configs
        )
        return env_ok and config_ok

    def get_steps(
        self, build_env: str, shard_id: int = 1, num_shards: int = 1
    ) -> list[TestStep]:
        return self.steps


# ---------------------------------------------------------------------------
# Concrete plan types
# ---------------------------------------------------------------------------


@dataclass
class CoreTestPlan(BasePytorchTestPlan):
    """Standard python test/run_test.py based tests.

    Use `steps` for simple plans. Use `get_steps_fn` when steps depend on
    shard_id / num_shards (or build_env) and must be generated at runtime.
    """

    get_steps_fn: Callable[[str, int, int, list[str]], list[TestStep]] | None = None

    def __post_init__(self) -> None:
        if bool(self.get_steps_fn) == bool(self.steps):
            raise ValueError(
                f"CoreTestPlan '{self.group_id}': provide exactly one of 'steps' or 'get_steps_fn'"
            )

    def get_steps(
        self, build_env: str, shard_id: int = 1, num_shards: int = 1
    ) -> list[TestStep]:
        if self.get_steps_fn:
            extra_args = resolve_extra_arg_list(self.extra_args, build_env)
            return self.get_steps_fn(build_env, shard_id, num_shards, extra_args)
        return self.steps
