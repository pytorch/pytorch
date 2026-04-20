"""
Unit tests for cli.lib.core.pytorch.lib and pytorch_test_library.

Testable on macOS without GPU or PyTorch installed — all subprocess calls
and side-effectful helpers are mocked.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli.lib.core.pytorch.pytorch_test_library import (
    BasePytorchTestPlan,
    BenchmarkTestPlan,
    CoreTestPlan,
    TestStep,
    is_cpu_only,
    is_cuda,
    is_gpu,
    is_rocm,
    is_xpu,
    matches_env,
    resolve_env_vars,
)

MOD = "cli.lib.core.pytorch.lib"


# ---------------------------------------------------------------------------
# Fixture: patch run helpers inside lib.py
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_lib(monkeypatch):
    """
    Patch side-effectful helpers in lib.py and expose them for assertions.
    """
    module = importlib.import_module(MOD)

    pip_install_packages = MagicMock(name="pip_install_packages")
    run_command = MagicMock(name="run_command", return_value=0)

    temp_calls: list[dict] = []
    workdir_calls: list[str] = []

    def fake_temp_env(mapping: dict[str, str]):
        temp_calls.append(mapping)
        return nullcontext()

    def fake_working_directory(path: str):
        workdir_calls.append(path)
        return nullcontext()

    logger = SimpleNamespace(
        info=MagicMock(name="logger.info"),
        warning=MagicMock(name="logger.warning"),
        error=MagicMock(name="logger.error"),
    )

    monkeypatch.setattr(module, "pip_install_packages", pip_install_packages, raising=True)
    monkeypatch.setattr(module, "run_command", run_command, raising=True)
    monkeypatch.setattr(module, "temp_environ", fake_temp_env, raising=True)
    monkeypatch.setattr(module, "working_directory", fake_working_directory, raising=True)
    monkeypatch.setattr(module, "logger", logger, raising=True)

    return SimpleNamespace(
        module=module,
        run_test_plan=module.run_test_plan,
        resolve_plan_for_test_config=module.resolve_plan_for_test_config,
        pip_install_packages=pip_install_packages,
        run_command=run_command,
        temp_calls=temp_calls,
        workdir_calls=workdir_calls,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Environment condition helpers
# ---------------------------------------------------------------------------


class TestEnvConditions:
    def test_is_cuda_matches_cuda_without_rocm(self):
        assert is_cuda("pytorch-linux-focal-cuda12.1-py3.10")
        assert not is_cuda("pytorch-linux-focal-rocm6.0-py3.10")
        assert not is_cuda("pytorch-linux-focal-py3.10")

    def test_is_rocm(self):
        assert is_rocm("pytorch-linux-focal-rocm6.0")
        assert not is_rocm("pytorch-linux-focal-cuda12.1")

    def test_is_xpu(self):
        assert is_xpu("pytorch-linux-focal-xpu-py3.10")
        assert not is_xpu("pytorch-linux-focal-cuda12.1")

    def test_is_gpu_covers_all_gpu_types(self):
        assert is_gpu("cuda12.1-py3.10")
        assert is_gpu("rocm6.0-py3.10")
        assert is_gpu("xpu-py3.10")
        assert not is_gpu("cpuonly-py3.10")

    def test_is_cpu_only_excludes_gpu(self):
        assert is_cpu_only("cpuonly-py3.10")
        assert not is_cpu_only("cuda12.1-py3.10")
        assert not is_cpu_only("rocm6.0-py3.10")

    def test_matches_env_string_is_substring(self):
        assert matches_env("cuda", "pytorch-linux-focal-cuda12.1")
        assert not matches_env("rocm", "pytorch-linux-focal-cuda12.1")

    def test_matches_env_callable(self):
        assert matches_env(is_cuda, "pytorch-linux-focal-cuda12.1")
        assert not matches_env(is_cuda, "pytorch-linux-focal-rocm6.0")


class TestResolveEnvVars:
    def test_static_dict(self):
        assert resolve_env_vars({"A": "1"}, "some-env") == {"A": "1"}

    def test_callable_receives_build_env(self):
        fn = lambda env: {"DEVICE": "hip"} if "rocm" in env else {"DEVICE": "cuda"}
        assert resolve_env_vars(fn, "rocm6.0") == {"DEVICE": "hip"}
        assert resolve_env_vars(fn, "cuda12.1") == {"DEVICE": "cuda"}


# ---------------------------------------------------------------------------
# BasePytorchTestPlan.is_eligible
# ---------------------------------------------------------------------------


class TestIsEligible:
    def _make_plan(self, run_on=None, test_configs=None):
        return BasePytorchTestPlan(
            group_id="test_plan",
            title="Test",
            steps=[],
            run_on=run_on or [],
            test_configs=test_configs or [],
        )

    def test_empty_conditions_always_eligible(self):
        plan = self._make_plan()
        assert plan.is_eligible("any-env", "any-config")

    def test_run_on_string_match(self):
        plan = self._make_plan(run_on=["cuda"])
        assert plan.is_eligible("pytorch-cuda12.1")
        assert not plan.is_eligible("pytorch-cpuonly")

    def test_run_on_callable(self):
        plan = self._make_plan(run_on=[is_cpu_only])
        assert plan.is_eligible("pytorch-cpuonly")
        assert not plan.is_eligible("pytorch-cuda12.1")

    def test_test_configs_filter(self):
        plan = self._make_plan(test_configs=["nogpu_NO_AVX2", "nogpu_AVX512"])
        assert plan.is_eligible("any-env", "nogpu_NO_AVX2")
        assert plan.is_eligible("any-env", "nogpu_AVX512")
        assert not plan.is_eligible("any-env", "default")
        assert not plan.is_eligible("any-env", "")

    def test_both_conditions_must_pass(self):
        plan = self._make_plan(run_on=["cuda"], test_configs=["default"])
        assert plan.is_eligible("pytorch-cuda12.1", "default")
        assert not plan.is_eligible("pytorch-cpuonly", "default")
        assert not plan.is_eligible("pytorch-cuda12.1", "nogpu_NO_AVX2")

    def test_run_on_any_matches(self):
        plan = self._make_plan(run_on=["cuda", "rocm"])
        assert plan.is_eligible("pytorch-cuda12.1")
        assert plan.is_eligible("pytorch-rocm6.0")
        assert not plan.is_eligible("pytorch-cpuonly")


# ---------------------------------------------------------------------------
# resolve_plan_for_test_config
# ---------------------------------------------------------------------------


def _make_library(*plans):
    return {p.group_id: p for p in plans}


class TestResolvePlanForTestConfig:
    def test_raises_when_build_env_missing(self, patch_lib):
        with pytest.raises(RuntimeError, match="build_env is required"):
            patch_lib.resolve_plan_for_test_config("default", "", {})

    def test_exact_match(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_cpuonly",
                title="CPU",
                steps=[],
                run_on=[is_cpu_only],
                test_configs=["nogpu_NO_AVX2"],
            ),
            CoreTestPlan(
                group_id="plan_gpu",
                title="GPU",
                steps=[],
                run_on=[is_gpu],
                test_configs=["default"],
            ),
        )
        result = patch_lib.resolve_plan_for_test_config("nogpu_NO_AVX2", "cpuonly-py3.10", lib)
        assert result == "plan_cpuonly"

    def test_no_match_raises(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_cpuonly",
                title="CPU",
                steps=[],
                test_configs=["nogpu_NO_AVX2"],
            )
        )
        with pytest.raises(RuntimeError, match="No plan matched"):
            patch_lib.resolve_plan_for_test_config("default", "cpuonly-py3.10", lib)

    def test_ambiguous_match_raises(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(group_id="plan_a", title="A", steps=[], test_configs=["default"]),
            CoreTestPlan(group_id="plan_b", title="B", steps=[], test_configs=["default"]),
        )
        with pytest.raises(RuntimeError, match="Ambiguous"):
            patch_lib.resolve_plan_for_test_config("default", "cuda12.1", lib)


# ---------------------------------------------------------------------------
# run_test_plan — basic execution
# ---------------------------------------------------------------------------


def _simple_step(calls: list, name: str = "step1") -> TestStep:
    def fn():
        calls.append(name)
    return TestStep(test_id=name, fn=fn)


class TestRunTestPlan:
    def test_raises_when_build_env_missing(self, patch_lib):
        lib = _make_library(CoreTestPlan(group_id="plan_a", title="A", steps=[]))
        with pytest.raises(RuntimeError, match="build_env is required"):
            patch_lib.run_test_plan("plan_a", "", library=lib)

    def test_raises_on_unknown_group(self, patch_lib):
        with pytest.raises(RuntimeError, match="not found"):
            patch_lib.run_test_plan("nonexistent", "some-env", library={})

    def test_runs_all_steps(self, patch_lib):
        calls: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[_simple_step(calls, "s1"), _simple_step(calls, "s2")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert calls == ["s1", "s2"]

    def test_runs_single_step_when_test_id_given(self, patch_lib):
        calls: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[_simple_step(calls, "s1"), _simple_step(calls, "s2")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", test_id="s2", library=lib)
        assert calls == ["s2"]

    def test_raises_on_unknown_test_id(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[_simple_step([], "s1")],
            )
        )
        with pytest.raises(RuntimeError, match="test_id 'nope' not found"):
            patch_lib.run_test_plan("plan_a", "some-env", test_id="nope", library=lib)

    def test_cmd_without_test_id_raises(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(group_id="plan_a", title="A", steps=[_simple_step([], "s1")])
        )
        with pytest.raises(RuntimeError, match="--cmd requires --test-id"):
            patch_lib.run_test_plan("plan_a", "some-env", cmd="pytest foo.py", library=lib)

    def test_cmd_replaces_step_fn(self, patch_lib):
        fn_called = []

        def fn():
            fn_called.append(True)

        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[TestStep(test_id="s1", fn=fn)],
            )
        )
        patch_lib.run_test_plan(
            "plan_a", "some-env", test_id="s1", cmd="pytest -k foo", library=lib
        )
        assert not fn_called
        patch_lib.run_command.assert_called_once_with("pytest -k foo", use_shell=True)

    def test_aggregates_failures_and_raises(self, patch_lib):
        def fail():
            raise RuntimeError("boom")

        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(test_id="ok", fn=lambda: None),
                    TestStep(test_id="bad1", fn=fail),
                    TestStep(test_id="bad2", fn=fail),
                ],
            )
        )
        with pytest.raises(RuntimeError, match="2 step"):
            patch_lib.run_test_plan("plan_a", "some-env", library=lib)

    def test_continues_after_step_failure(self, patch_lib):
        calls: list[str] = []

        def fail():
            raise RuntimeError("boom")

        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(test_id="s1", fn=fail),
                    TestStep(test_id="s2", fn=lambda: calls.append("s2")),
                ],
            )
        )
        with pytest.raises(RuntimeError):
            patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert "s2" in calls


# ---------------------------------------------------------------------------
# run_test_plan — env vars and working_dir
# ---------------------------------------------------------------------------


class TestRunTestPlanEnvVars:
    def test_plan_level_env_applied_once(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                env_vars={"PLAN_VAR": "1"},
                steps=[_simple_step([], "s1")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        # First temp_environ call is the plan-level one
        assert patch_lib.temp_calls[0] == {"PLAN_VAR": "1"}

    def test_step_level_env_applied_per_step(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(test_id="s1", fn=lambda: None, env_vars={"STEP_VAR": "a"}),
                    TestStep(test_id="s2", fn=lambda: None, env_vars={"STEP_VAR": "b"}),
                ],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        step_envs = patch_lib.temp_calls[1:]  # skip plan-level call at index 0
        assert {"STEP_VAR": "a"} in step_envs
        assert {"STEP_VAR": "b"} in step_envs

    def test_callable_env_vars_resolved_with_build_env(self, patch_lib):
        env_fn = lambda env: {"DEVICE": "hip"} if "rocm" in env else {"DEVICE": "cuda"}
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                env_vars=env_fn,
                steps=[_simple_step([], "s1")],
            )
        )
        patch_lib.run_test_plan("plan_a", "pytorch-rocm6.0", library=lib)
        assert patch_lib.temp_calls[0] == {"DEVICE": "hip"}

    def test_working_dir_from_step(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[TestStep(test_id="s1", fn=lambda: None, working_dir="src/tests")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert "src/tests" in patch_lib.workdir_calls

    def test_step_working_dir_overrides_plan(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                working_dir="plan_dir",
                steps=[TestStep(test_id="s1", fn=lambda: None, working_dir="step_dir")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert "step_dir" in patch_lib.workdir_calls
        assert "plan_dir" not in patch_lib.workdir_calls


# ---------------------------------------------------------------------------
# run_test_plan — setup_fn
# ---------------------------------------------------------------------------


class TestRunTestPlanSetupFn:
    def test_plan_setup_fn_always_runs(self, patch_lib):
        setup_calls: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                setup_fn=lambda: setup_calls.append("plan_setup"),
                steps=[_simple_step([], "s1")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert setup_calls == ["plan_setup"]

    def test_plan_setup_fn_runs_for_single_step_repro(self, patch_lib):
        setup_calls: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                setup_fn=lambda: setup_calls.append("plan_setup"),
                steps=[_simple_step([], "s1"), _simple_step([], "s2")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", test_id="s1", library=lib)
        assert setup_calls == ["plan_setup"]

    def test_step_setup_fn_runs_before_step_fn(self, patch_lib):
        order: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(
                        test_id="s1",
                        fn=lambda: order.append("fn"),
                        setup_fn=lambda: order.append("setup"),
                    )
                ],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert order == ["setup", "fn"]

    def test_step_setup_fn_runs_with_cmd(self, patch_lib):
        setup_calls: list[str] = []
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(
                        test_id="s1",
                        fn=lambda: None,
                        setup_fn=lambda: setup_calls.append("step_setup"),
                    )
                ],
            )
        )
        patch_lib.run_test_plan(
            "plan_a", "some-env", test_id="s1", cmd="pytest foo.py", library=lib
        )
        assert setup_calls == ["step_setup"]


# ---------------------------------------------------------------------------
# run_test_plan — pip_installs
# ---------------------------------------------------------------------------


class TestRunTestPlanPipInstalls:
    def test_plan_level_pip_installs(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                pip_installs=[["timm==1.0.0"], ["-e", "."]],
                steps=[_simple_step([], "s1")],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        calls = patch_lib.pip_install_packages.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == ["timm==1.0.0"]
        assert calls[1].args[0] == ["-e", "."]

    def test_step_level_pip_installs(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                steps=[
                    TestStep(
                        test_id="s1",
                        fn=lambda: None,
                        pip_installs=[["flash-attn"]],
                    )
                ],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        patch_lib.pip_install_packages.assert_called_once_with(["flash-attn"])

    def test_plan_and_step_pip_installs_combined(self, patch_lib):
        lib = _make_library(
            CoreTestPlan(
                group_id="plan_a",
                title="A",
                pip_installs=[["pkg-a"]],
                steps=[
                    TestStep(test_id="s1", fn=lambda: None, pip_installs=[["pkg-b"]])
                ],
            )
        )
        patch_lib.run_test_plan("plan_a", "some-env", library=lib)
        assert patch_lib.pip_install_packages.call_count == 2


# ---------------------------------------------------------------------------
# BenchmarkTestPlan
# ---------------------------------------------------------------------------


class TestBenchmarkTestPlan:
    def test_post_init_builds_steps_from_models(self):
        plan = BenchmarkTestPlan(
            group_id="bench_a",
            title="Bench",
            steps=[],
            models=["resnet50", "BERT_pytorch"],
            modes=["training"],
        )
        step_ids = [s.test_id for s in plan.steps]
        assert "resnet50" in step_ids
        assert "BERT_pytorch" in step_ids
        assert "join_results" in step_ids
        assert step_ids[-1] == "join_results"

    def test_per_model_output_path_format(self):
        plan = BenchmarkTestPlan(
            group_id="bench_a",
            title="Bench",
            steps=[],
            backend="inductor",
            suite="torchbench",
            device="cuda",
            modes=["training"],
            models=["resnet50"],
            output_dir="test/test-reports",
        )
        path = plan._per_model_output_path("resnet50")
        assert "inductor" in path
        assert "torchbench" in path
        assert "resnet50" in path
        assert "training" in path
        assert "cuda" in path
        assert path.startswith("test/test-reports")

    def testjoin_results_called_with_all_paths(self, patch_lib):
        joined: list[list[str]] = []
        plan = BenchmarkTestPlan(
            group_id="bench_a",
            title="Bench",
            steps=[],
            run_on=[is_cuda],
            models=["m1", "m2"],
            modes=["training"],
            join_results_fn=lambda paths: joined.append(paths),
        )
        lib = {plan.group_id: plan}

        # Mock _run_model so it doesn't call real subprocess
        with (
            MagicMock() as mock_run,
        ):
            for step in plan.steps[:-1]:  # all except join_results
                step.fn = lambda: None

            patch_lib.run_test_plan("bench_a", "pytorch-cuda12.1", library=lib)

        assert len(joined) == 1
        assert len(joined[0]) == 2

    def test_single_model_reproduction(self, patch_lib):
        run_calls: list[str] = []

        plan = BenchmarkTestPlan(
            group_id="bench_a",
            title="Bench",
            steps=[],
            run_on=[is_cuda],
            models=["m1", "m2"],
            modes=["training"],
        )
        lib = {plan.group_id: plan}

        # Replace step fns with trackers
        for step in plan.steps:
            name = step.test_id
            step.fn = lambda n=name: run_calls.append(n)

        patch_lib.run_test_plan("bench_a", "pytorch-cuda12.1", test_id="m1", library=lib)
        assert run_calls == ["m1"]
