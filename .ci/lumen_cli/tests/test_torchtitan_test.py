from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call


def test_prepare_installs_built_dependency_wheels_without_deps(monkeypatch):
    module = importlib.import_module("cli.lib.core.torchtitan.torchtitan_test")

    clone_torchtitan = MagicMock(name="clone_torchtitan")
    first_matching_pkg = MagicMock(
        name="first_matching_pkg",
        side_effect=[
            "dist/ao/torchao-test.whl",
            "dist/torchcomms/torchcomms-test.whl",
        ],
    )
    pip_install_packages = MagicMock(name="pip_install_packages")

    monkeypatch.setattr(module, "clone_torchtitan", clone_torchtitan)
    monkeypatch.setattr(module, "first_matching_pkg", first_matching_pkg)
    monkeypatch.setattr(module, "pip_install_packages", pip_install_packages)
    monkeypatch.setattr(module, "working_directory", lambda _: nullcontext())

    runner = module.TorchtitanTestRunner(
        SimpleNamespace(test_plan="torchtitan_features_integration")
    )
    runner.prepare()

    clone_torchtitan.assert_called_once_with(dst="torchtitan")
    first_matching_pkg.assert_has_calls(
        [
            call("dist/ao/torchao*.whl"),
            call("dist/torchcomms/torchcomms*.whl"),
        ]
    )
    assert pip_install_packages.call_args_list[0] == call(
        packages=[
            "--no-index",
            "--no-deps",
            "dist/ao/torchao-test.whl",
            "dist/torchcomms/torchcomms-test.whl",
        ]
    )
