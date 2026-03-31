"""Unit tests for cli.lib.pytorch.runner and base."""
# ruff: noqa: S101

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cli.lib.pytorch.base import (
    BasePytorchTestPlan,
    CoreTestPlan,
    is_cpu_only,
    is_cuda,
    is_gpu,
    is_xpu,
    matches_env,
    resolve_env_vars,
    resolve_extra_arg_list,
    TestStep,
)


MOD = "cli.lib.pytorch.runner"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(name: str = "s1", calls: list | None = None) -> TestStep:
    def fn():
        if calls is not None:
            calls.append(name)

    return TestStep(test_id=name, fn=fn)


def _plan(*steps, group_id="plan_a", **kwargs) -> CoreTestPlan:
    return CoreTestPlan(group_id=group_id, title="T", steps=list(steps), **kwargs)


def _lib(*plans) -> dict:
    return {p.group_id: p for p in plans}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_lib(monkeypatch):
    module = importlib.import_module(MOD)

    pip_mock = MagicMock(name="pip_install_packages")
    run_mock = MagicMock(name="run_command", return_value=0)
    temp_calls: list[dict] = []

    monkeypatch.setattr(module, "pip_install_packages", pip_mock, raising=True)
    monkeypatch.setattr(module, "run_command", run_mock, raising=True)
    monkeypatch.setattr(
        module,
        "temp_environ",
        lambda m: (temp_calls.append(m), nullcontext())[1],
        raising=True,
    )
    monkeypatch.setattr(
        module, "working_directory", lambda p: nullcontext(), raising=True
    )
    logger_mock = SimpleNamespace(
        info=MagicMock(), warning=MagicMock(), error=MagicMock()
    )
    monkeypatch.setattr(module, "logger", logger_mock, raising=True)

    ns = SimpleNamespace(
        run_test_plan=module.run_test_plan,
        resolve_plans_for_test_config=module.resolve_plans_for_test_config,
        pip=pip_mock,
        run=run_mock,
        temp_calls=temp_calls,
        logger=logger_mock,
    )
    return ns


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


class TestEnvHelpers:
    def test_is_cuda_excludes_rocm(self):
        assert is_cuda("cuda12.1")
        assert not is_cuda("rocm6.0")

    def test_is_gpu(self):
        assert is_gpu("cuda12.1")
        assert is_gpu("rocm6.0")
        assert not is_gpu("cpuonly")

    def test_is_cpu_only(self):
        assert is_cpu_only("cpuonly")
        assert not is_cpu_only("cuda12.1")

    def test_matches_env_string(self):
        assert matches_env("cuda", "pytorch-cuda12.1")
        assert not matches_env("rocm", "pytorch-cuda12.1")

    def test_resolve_env_vars_callable(self):
        def fn(env):
            return {"D": "hip"} if "rocm" in env else {"D": "cuda"}

        assert resolve_env_vars(fn, "rocm6.0") == {"D": "hip"}
        assert resolve_env_vars(fn, "cuda12.1") == {"D": "cuda"}


# ---------------------------------------------------------------------------
# resolve_extra_arg_list
# ---------------------------------------------------------------------------


class TestResolveExtraArgList:
    def test_static_string(self):
        assert resolve_extra_arg_list(["--verbose"], "env") == ["--verbose"]

    def test_callable_returns_value(self):
        assert resolve_extra_arg_list(
            [lambda env: "--xpu" if is_xpu(env) else None], "linux-xpu"
        ) == ["--xpu"]

    def test_callable_returns_none_skipped(self):
        assert (
            resolve_extra_arg_list(
                [lambda env: "--xpu" if is_xpu(env) else None], "linux-cuda"
            )
            == []
        )

    def test_empty(self):
        assert resolve_extra_arg_list([], "env") == []

    def test_mixed(self):
        result = resolve_extra_arg_list(
            ["--flag", lambda env: "--xpu" if is_xpu(env) else None],
            "linux-xpu",
        )
        assert result == ["--flag", "--xpu"]


# ---------------------------------------------------------------------------
# is_eligible
# ---------------------------------------------------------------------------


class TestIsEligible:
    def test_empty_always_eligible(self):
        plan = BasePytorchTestPlan(group_id="p", title="T", steps=[])
        assert plan.is_eligible("any-env", "any-config")

    def test_run_on_and_test_config_both_required(self):
        plan = BasePytorchTestPlan(
            group_id="p",
            title="T",
            steps=[],
            run_on=["cuda"],
            test_configs=["default"],
        )
        assert plan.is_eligible("cuda12.1", "default")
        assert not plan.is_eligible("cpuonly", "default")
        assert not plan.is_eligible("cuda12.1", "other")

    def test_run_on_callable(self):
        plan = BasePytorchTestPlan(
            group_id="p", title="T", steps=[], run_on=[is_cpu_only]
        )
        assert plan.is_eligible("cpuonly")
        assert not plan.is_eligible("cuda12.1")


# ---------------------------------------------------------------------------
# CoreTestPlan — steps vs get_steps_fn
# ---------------------------------------------------------------------------


class TestCoreTestPlanSteps:
    def test_steps_valid(self):
        plan = _plan(_step())
        assert plan.get_steps("env") == plan.steps

    def test_get_steps_fn_valid(self):
        s = _step()
        plan = CoreTestPlan(
            group_id="p", title="T", get_steps_fn=lambda e, si, ns, ea: [s]
        )
        assert plan.get_steps("env") == [s]

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            CoreTestPlan(group_id="p", title="T")

    def test_both_raises(self):
        s = _step()
        with pytest.raises(ValueError, match="exactly one"):
            CoreTestPlan(
                group_id="p",
                title="T",
                steps=[s],
                get_steps_fn=lambda e, si, ns, ea: [s],
            )

    def test_shard_args_passed_to_fn(self):
        received = {}

        def fn(build_env, shard_id, num_shards, extra_args):
            received.update(shard_id=shard_id, num_shards=num_shards)
            return [_step()]

        plan = CoreTestPlan(group_id="p", title="T", get_steps_fn=fn)
        plan.get_steps("env", shard_id=2, num_shards=4)
        assert received == {"shard_id": 2, "num_shards": 4}

    def test_shard_passed_through_run_test_plan(self, patch_lib):
        received = {}

        def fn(build_env, shard_id, num_shards, extra_args):
            received.update(shard_id=shard_id, num_shards=num_shards)
            return [_step()]

        lib = {"p": CoreTestPlan(group_id="p", title="T", get_steps_fn=fn)}
        patch_lib.run_test_plan("p", "env", shard_id=3, num_shards=8, library=lib)
        assert received == {"shard_id": 3, "num_shards": 8}


# ---------------------------------------------------------------------------
# resolve_plans_for_test_config
# ---------------------------------------------------------------------------


class TestResolvePlan:
    def test_raises_when_no_build_env(self, patch_lib):
        with pytest.raises(RuntimeError, match="build_env is required"):
            patch_lib.resolve_plans_for_test_config("default", "", {})

    def test_single_match(self, patch_lib):
        lib = _lib(
            _plan(
                _step(), group_id="cpu", run_on=[is_cpu_only], test_configs=["nogpu"]
            ),
            _plan(_step(), group_id="gpu", run_on=[is_gpu], test_configs=["default"]),
        )
        assert patch_lib.resolve_plans_for_test_config("nogpu", "cpuonly", lib) == [
            "cpu"
        ]

    def test_multiple_match_runs_all(self, patch_lib):
        # Two plans with same test_config both match — returned in registry order.
        lib = _lib(
            _plan(_step(), group_id="a", test_configs=["default"]),
            _plan(_step(), group_id="b", test_configs=["default"]),
        )
        assert patch_lib.resolve_plans_for_test_config("default", "cuda12.1", lib) == [
            "a",
            "b",
        ]

    def test_no_match_raises(self, patch_lib):
        lib = _lib(_plan(_step(), group_id="p", test_configs=["nogpu"]))
        with pytest.raises(RuntimeError, match="No plan matched"):
            patch_lib.resolve_plans_for_test_config("default", "cpuonly", lib)


# ---------------------------------------------------------------------------
# run_test_plan
# ---------------------------------------------------------------------------


class TestRunTestPlan:
    def test_raises_when_no_build_env(self, patch_lib):
        with pytest.raises(RuntimeError, match="build_env is required"):
            patch_lib.run_test_plan("p", "", library=_lib(_plan(_step())))

    def test_raises_on_unknown_group(self, patch_lib):
        with pytest.raises(RuntimeError, match="not found"):
            patch_lib.run_test_plan("nope", "env", library={})

    def test_runs_all_steps(self, patch_lib):
        calls: list[str] = []
        lib = _lib(_plan(_step("s1", calls), _step("s2", calls)))
        patch_lib.run_test_plan("plan_a", "env", library=lib)
        assert calls == ["s1", "s2"]

    def test_filter_by_test_id(self, patch_lib):
        calls: list[str] = []
        lib = _lib(_plan(_step("s1", calls), _step("s2", calls)))
        patch_lib.run_test_plan("plan_a", "env", test_id="s2", library=lib)
        assert calls == ["s2"]

    def test_cmd_replaces_fn(self, patch_lib):
        called = []
        lib = _lib(_plan(TestStep(test_id="s1", fn=lambda: called.append(True))))
        patch_lib.run_test_plan(
            "plan_a", "env", test_id="s1", cmd="pytest foo.py", library=lib
        )
        assert not called
        patch_lib.run.assert_called_once_with("pytest foo.py", use_shell=True)

    def test_cmd_without_test_id_raises(self, patch_lib):
        with pytest.raises(RuntimeError, match="--cmd requires --test-id"):
            patch_lib.run_test_plan(
                "plan_a", "env", cmd="pytest", library=_lib(_plan(_step()))
            )

    def test_continues_after_failure_and_raises(self, patch_lib):
        calls: list[str] = []
        lib = _lib(
            _plan(
                TestStep(
                    test_id="bad",
                    fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                ),
                _step("s2", calls),
            )
        )
        with pytest.raises(RuntimeError, match="1 step"):
            patch_lib.run_test_plan("plan_a", "env", library=lib)
        assert "s2" in calls

    def test_plan_env_vars_applied(self, patch_lib):
        lib = _lib(_plan(_step(), env_vars={"K": "v"}))
        patch_lib.run_test_plan("plan_a", "env", library=lib)
        assert patch_lib.temp_calls[0] == {"K": "v"}

    def test_setup_fn_runs_before_steps(self, patch_lib):
        order: list[str] = []
        lib = _lib(
            _plan(
                TestStep(test_id="s1", fn=lambda: order.append("step")),
                setup_fn=lambda: order.append("setup"),
            )
        )
        patch_lib.run_test_plan("plan_a", "env", library=lib)
        assert order == ["setup", "step"]

    def test_hardware_mismatch_warns(self, patch_lib):
        # group_id specified explicitly with mismatched build_env → warning, still runs.
        lib = _lib(_plan(_step(), group_id="plan_a", run_on=[is_gpu]))
        patch_lib.run_test_plan("plan_a", "cpuonly", library=lib)
        assert patch_lib.logger.warning.called


# ---------------------------------------------------------------------------
# LUMEN_DRY_RUN
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_and_skips(self, monkeypatch, capsys):
        import cli.lib.pytorch.base as base_mod

        monkeypatch.setenv("LUMEN_DRY_RUN", "1")
        run_mock = MagicMock()
        monkeypatch.setattr(base_mod, "run_command", run_mock)
        base_mod.run_test("--include", "test_foo")
        run_mock.assert_not_called()
        assert "[dry-run]" in capsys.readouterr().out

    def test_no_dry_run_executes(self, monkeypatch):
        import cli.lib.pytorch.base as base_mod

        monkeypatch.delenv("LUMEN_DRY_RUN", raising=False)
        run_mock = MagicMock()
        monkeypatch.setattr(base_mod, "run_command", run_mock)
        base_mod.run_test("--include", "test_foo")
        run_mock.assert_called_once()
