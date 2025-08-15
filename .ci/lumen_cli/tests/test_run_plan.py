# tests/test_run_test_plan.py
import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import call, MagicMock

import pytest


MOD = "cli.lib.core.vllm.lib"

# We import inside tests so the MOD override above applies everywhere
run_test_plan_import_path = f"{MOD}.run_test_plan"


@pytest.fixture
def patch_module(monkeypatch):
    """
    Patch helpers ('pip_install_packages', 'temp_environ', 'working_directory',
    'run_command', 'logger') inside the target module and expose them.
    """
    module = importlib.import_module(MOD)

    # Create fakes/mocks
    pip_install_packages = MagicMock(name="pip_install_packages")
    run_command = MagicMock(name="run_command", return_value=0)

    # temp_environ / working_directory: record calls but act as context managers
    temp_calls: list[dict] = []
    workdir_calls: list[str] = []

    def fake_temp_environ(env: dict):
        temp_calls.append(env)
        return nullcontext()

    def fake_working_directory(path: str):
        workdir_calls.append(path)
        return nullcontext()

    logger = SimpleNamespace(
        info=MagicMock(name="logger.info"),
        error=MagicMock(name="logger.error"),
    )

    # Apply patches (raise if attribute doesn't exist)
    monkeypatch.setattr(
        module, "pip_install_packages", pip_install_packages, raising=True
    )
    monkeypatch.setattr(module, "run_command", run_command, raising=True)
    monkeypatch.setattr(module, "temp_environ", fake_temp_environ, raising=True)
    monkeypatch.setattr(
        module, "working_directory", fake_working_directory, raising=True
    )
    monkeypatch.setattr(module, "logger", logger, raising=True)

    return SimpleNamespace(
        module=module,
        run_test_plan=module.run_test_plan,  # expose to avoid getattr("constant") (Ruff B009)
        pip_install_packages=pip_install_packages,
        run_command=run_command,
        temp_calls=temp_calls,
        workdir_calls=workdir_calls,
        logger=logger,
    )


def test_success_runs_all_steps_and_uses_env_and_workdir(monkeypatch, patch_module):
    run_test_plan = patch_module.run_test_plan

    tests_map = {
        "basic": {
            "title": "Basic suite",
            "package_install": [],
            "env_var": {"GLOBAL_FLAG": "1"},
            "working_directory": "tests",
            "steps": [
                {"command": "pytest -q", "env_var": {"A": "x"}},
                {"command": "pytest -q tests/unit", "env_var": {"B": "y"}},
            ],
        }
    }

    # All steps succeed
    patch_module.run_command.side_effect = [0, 0]

    run_test_plan("basic", "cpu", tests_map)

    # Called twice with our commands, check= False
    assert patch_module.run_command.call_args_list == [
        call(cmd="pytest -q", check=False),
        call(cmd="pytest -q tests/unit", check=False),
    ]

    # Outer env + two step env context entries, in order
    assert patch_module.temp_calls == [
        {"GLOBAL_FLAG": "1"},  # outer
        {"A": "x"},  # step 1
        {"B": "y"},  # step 2
    ]

    # Working directory (defaults to "tests" already specified)
    assert patch_module.workdir_calls == ["tests"]


def test_installs_packages_when_present(monkeypatch, patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "with_pkgs": {
            "title": "Needs deps",
            "package_install": ["timm==1.0.0", "flash-attn"],
            "steps": [{"command": "pytest -q"}],
        }
    }

    patch_module.run_command.return_value = 0

    run_test_plan("with_pkgs", "gpu", tests_map)

    patch_module.pip_install_packages.assert_called_once_with(
        packages=["timm==1.0.0", "flash-attn"],
        prefer_uv=True,
    )


def test_raises_on_missing_plan(patch_module):
    run_test_plan = patch_module.module.run_test_plan
    with pytest.raises(RuntimeError) as ei:
        run_test_plan("nope", "cpu", tests_map={})

    assert "test nope not found" in str(ei.value)


def test_aggregates_failures_and_raises(monkeypatch, patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "mix": {
            "title": "Some pass some fail",
            "steps": [
                {"command": "pytest test_a.py"},  # 0 → pass
                {"command": "pytest test_b.py"},  # 1 → fail
                {"command": "pytest test_c.py"},  # 2 → fail
            ],
        }
    }

    # Simulate pass, fail, fail
    patch_module.run_command.side_effect = [0, 1, 2]

    with pytest.raises(RuntimeError) as ei:
        run_test_plan("mix", "cpu", tests_map)

    msg = str(ei.value)
    assert "2 pytest runs failed" in msg
    # Ensure logger captured failed tests list
    patch_module.logger.error.assert_called_once()
    # And we attempted all three commands
    assert patch_module.run_command.call_count == 3


def test_custom_working_directory_used(patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "customwd": {
            "title": "Custom wd",
            "working_directory": "examples/ci",
            "steps": [{"command": "pytest -q"}],
        }
    }

    patch_module.run_command.return_value = 0
    run_test_plan("customwd", "cpu", tests_map)

    assert patch_module.workdir_calls == ["examples/ci"]
