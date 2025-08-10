import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from cli.lib.core.vllm import VllmBuildRunner

import os
import types
import pytest

MODULE = "cli.lib.core.vllm_build"  # <-- CHANGE ME to your module path

@pytest.fixture
def mod(monkeypatch):
    """Import the module under test and return it (after we can monkeypatch)."""
    import importlib
    m = importlib.import_module(MODULE)
    return m

# ------------------------------
# VllmBuildInputs.__post_init__
# ------------------------------

def test_inputs_ok_when_flags_set_and_paths_exist(monkeypatch, tmp_path, mod):
    # arrange: create fake files/dirs
    torch_dir = tmp_path / "dist"
    torch_dir.mkdir()
    dockerfile = tmp_path / "Dockerfile.tmp_vllm"
    dockerfile.write_text("# dockerfile")
    # stubs
    monkeypatch.setattr(mod, "is_path_exist", lambda p: os.path.exists(p))
    monkeypatch.setattr(mod, "local_image_exists", lambda name: True)
    monkeypatch.setattr(mod, "get_abs_path", lambda p: os.path.abspath(p))

    # env (your dataclass reads env via get_env default_factory)
    monkeypatch.setenv("USE_TORCH_WHEEL", "1")
    monkeypatch.setenv("USE_LOCAL_DOCKERFILE", "1")
    monkeypatch.setenv("USE_LOCAL_BASE_IMAGE", "1")
    monkeypatch.setenv("TORCH_WHEELS_PATH", str(torch_dir))
    monkeypatch.setenv("DOCKERFILE_PATH", str(dockerfile))
    monkeypatch.setenv("BASE_IMAGE", "local/image:tag")

    inputs = mod.VllmBuildInputs()

    # assert: paths normalized to absolute
    assert inputs.torch_whls_path == os.path.abspath(str(torch_dir))
    assert inputs.dockerfile_path == os.path.abspath(str(dockerfile))
    assert inputs.base_image == "local/image:tag"

def test_inputs_raises_when_flag_on_but_missing_path(monkeypatch, mod):
    # torch wheel path check should raise
    monkeypatch.setattr(mod, "is_path_exist", lambda p: False)
    monkeypatch.setattr(mod, "local_image_exists", lambda name: True)
    monkeypatch.setattr(mod, "get_abs_path", lambda p: p)

    monkeypatch.setenv("USE_TORCH_WHEEL", "1")
    monkeypatch.setenv("TORCH_WHEELS_PATH", "dist-missing")
    # off for other checks to isolate the failure
    monkeypatch.setenv("USE_LOCAL_DOCKERFILE", "0")
    monkeypatch.setenv("USE_LOCAL_BASE_IMAGE", "0")

    with pytest.raises(FileNotFoundError):
        mod.VllmBuildInputs()

def test_inputs_skip_checks_when_flags_off(monkeypatch, mod):
    # No path functions should be called if flags are "0"
    called = {"path": 0, "img": 0}
    monkeypatch.setattr(mod, "is_path_exist", lambda p: called.__setitem__("path", called["path"]+1))
    monkeypatch.setattr(mod, "local_image_exists", lambda name: called.__setitem__("img", called["img"]+1))
    monkeypatch.setattr(mod, "get_abs_path", lambda p: p)

    monkeypatch.setenv("USE_TORCH_WHEEL", "0")
    monkeypatch.setenv("USE_LOCAL_DOCKERFILE", "0")
    monkeypatch.setenv("USE_LOCAL_BASE_IMAGE", "0")

    _ = mod.VllmBuildInputs()
    # no calls executed by checks loop
    assert called["path"] == 0
    assert called["img"] == 0

# ------------------------------
# VllmBuildRunner helpers
# ------------------------------

def test_get_torch_wheel_path_arg(mod):
    r = mod.VllmBuildRunner()
    assert r._get_torch_wheel_path_arg("") == ""
    assert r._get_torch_wheel_path_arg("some/dir") == "--build-arg TORCH_WHEELS_PATH=tmp"

def test_get_base_image_args_local(monkeypatch, mod):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(
        use_local_base_image="1",
        base_image="local/image:tag",
    )
    monkeypatch.setattr(mod, "local_image_exists", lambda name: True)
    base_arg, final_arg, pull = r._get_base_image_args(inputs)
    assert base_arg == "--build-arg BUILD_BASE_IMAGE=local/image:tag"
    assert final_arg == "--build-arg FINAL_BASE_IMAGE=local/image:tag"
    assert pull == "--pull=false"

def test_get_base_image_args_remote(monkeypatch, mod):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(
        use_local_base_image="1",
        base_image="remote/image:tag",
    )
    monkeypatch.setattr(mod, "local_image_exists", lambda name: False)
    base_arg, final_arg, pull = r._get_base_image_args(inputs)
    assert base_arg == "--build-arg BUILD_BASE_IMAGE=remote/image:tag"
    assert final_arg == "--build-arg FINAL_BASE_IMAGE=remote/image:tag"
    assert pull == ""  # no --pull=false when not found locally

def test_get_base_image_args_flag_off(mod):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(
        use_local_base_image="0",
        base_image="ignored",
    )
    base_arg, final_arg, pull = r._get_base_image_args(inputs)
    assert base_arg == final_arg == pull == ""

# ------------------------------
# cp_* helpers
# ------------------------------

def test_cp_torch_whls_if_exist_copies(monkeypatch, tmp_path, mod):
    r = mod.VllmBuildRunner()
    src = tmp_path / "dist"
    src.mkdir()
    inputs = types.SimpleNamespace(
        use_torch_whl="1",
        torch_whls_path=str(src),
    )
    monkeypatch.setattr(mod, "is_path_exist", lambda p: os.path.exists(p))
    monkeypatch.setattr(mod, "get_abs_path", lambda p: os.path.abspath(p))
    monkeypatch.setattr(mod, "force_create_dir", lambda p: None)

    captured = {}
    def fake_run_cmd(cmd, log_cmd=False, cwd=None, env=None):
        captured["cmd"] = cmd
    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    tmp_dir = r.cp_torch_whls_if_exist(inputs)
    assert tmp_dir == f"./{r.work_directory}/{mod._VLLM_TEMP_FOLDER}"
    assert "cp -a" in captured["cmd"]
    assert str(src) in captured["cmd"]

def test_cp_torch_whls_if_exist_returns_empty_when_flag_off(mod):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(use_torch_whl="0", torch_whls_path="x")
    assert r.cp_torch_whls_if_exist(inputs) == ""

def test_cp_dockerfile_if_exist_copies(monkeypatch, tmp_path, mod, capsys):
    r = mod.VllmBuildRunner()
    df = tmp_path / "Dockerfile.tmp_vllm"
    df.write_text("# dockerfile")
    inputs = types.SimpleNamespace(
        use_local_dockrfile="1",
        dockerfile_path=str(df),
    )
    monkeypatch.setattr(mod, "is_path_exist", lambda p: os.path.exists(p))
    monkeypatch.setattr(mod, "get_abs_path", lambda p: os.path.abspath(p))

    captured = {}
    monkeypatch.setattr(mod, "run_cmd", lambda cmd, **kw: captured.setdefault("cmd", cmd))

    r.cp_dockerfile_if_exist(inputs)
    assert "cp " in captured["cmd"]
    assert "Dockerfile.nightly_torch" in captured["cmd"]

def test_cp_dockerfile_if_exist_skip_when_flag_off(monkeypatch, mod, capsys):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(use_local_dockrfile="0", dockerfile_path="ignored")
    # should not raise or call run_cmd
    called = {"run": 0}
    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: called.__setitem__("run", 1))
    r.cp_dockerfile_if_exist(inputs)
    assert called["run"] == 0

def test_cp_dockerfile_if_exist_raises_when_missing(monkeypatch, mod):
    r = mod.VllmBuildRunner()
    inputs = types.SimpleNamespace(use_local_dockrfile="1", dockerfile_path="missing")
    monkeypatch.setattr(mod, "is_path_exist", lambda p: False)
    with pytest.raises(FileNotFoundError):
        r.cp_dockerfile_if_exist(inputs)

# ------------------------------
# _generate_docker_build_cmd
# ------------------------------

def test_generate_docker_build_cmd_includes_expected_flags(monkeypatch, mod):
    r = mod.VllmBuildRunner()

    # Freeze cfg values by overriding VllmDockerBuildArgs() instance
    class FakeCfg:
        output_dir = "shared"
        target = "export-wheels"
        tag_name = "vllm-wheels"
        cuda = "12.8.1"
        py = "3.12"
        max_jobs = "64"
        sccache_bucket = ""
        sccache_region = ""
        torch_cuda_arch_list = "8.0"

    monkeypatch.setattr(mod, "VllmDockerBuildArgs", lambda: FakeCfg)

    inputs = types.SimpleNamespace(
        use_local_base_image="0",
        base_image="",
    )

    cmd = r._generate_docker_build_cmd(inputs, "/abs/out", "")
    # Assertions on key pieces
    assert "docker buildx build" in cmd
    assert "--output type=local,dest=/abs/out" in cmd
    assert "-f docker/Dockerfile.nightly_torch" in cmd
    assert "--build-arg CUDA_VERSION=12.8.1" in cmd
    assert "--build-arg PYTHON_VERSION=3.12" in cmd
    assert "--build-arg max_jobs=64" in cmd
    assert "--build-arg torch_cuda_arch_list='8.0'" in cmd
    assert "--target export-wheels" in cmd
    assert "-t vllm-wheels" in cmd

def test_generate_docker_build_cmd_includes_base_image_args(monkeypatch, mod):
    r = mod.VllmBuildRunner()
    # make local_image_exists True path
    monkeypatch.setattr(mod, "local_image_exists", lambda _: True)
    inputs = types.SimpleNamespace(
        use_local_base_image="1",
        base_image="local/image:tag",
    )
    # Freeze cfg minimal
    class FakeCfg:
        output_dir = "shared"; target = "export-wheels"; tag_name = "t"; cuda = "12.8.1"
        py = "3.12"; max_jobs = "64"; sccache_bucket = ""; sccache_region = ""; torch_cuda_arch_list = "8.0"
    monkeypatch.setattr(mod, "VllmDockerBuildArgs", lambda: FakeCfg)

    cmd = r._generate_docker_build_cmd(inputs, "/abs/out", "tmp")
    assert "--build-arg BUILD_BASE_IMAGE=local/image:tag" in cmd
    assert "--build-arg FINAL_BASE_IMAGE=local/image:tag" in cmd
    assert "--pull=false" in cmd
    assert "--build-arg TORCH_WHEELS_PATH=tmp" in cmd  # via _get_torch_wheel_path_arg("tmp")

# ------------------------------
# VllmBuildRunner.run (smoke)
# ------------------------------

def test_run_smoke(monkeypatch, tmp_path, mod):
    # Arrange: make all file ops and external calls no-op
    monkeypatch.setenv("USE_TORCH_WHEEL", "0")
    monkeypatch.setenv("USE_LOCAL_DOCKERFILE", "0")
    monkeypatch.setenv("USE_LOCAL_BASE_IMAGE", "0")
    monkeypatch.setenv("output_dir", str(tmp_path))

    monkeypatch.setattr(mod, "clone_vllm", lambda: None)
    monkeypatch.setattr(mod, "ensure_dir_exists", lambda p: None)
    monkeypatch.setattr(mod, "get_abs_path", lambda p: os.path.abspath(p))
    monkeypatch.setattr(mod.VllmBuildRunner, "_generate_docker_build_cmd", lambda self, i, out, w: "echo ok")
    called = {"run_cmd": 0}
    monkeypatch.setattr(mod, "run_cmd", lambda cmd, **kw: called.__setitem__("run_cmd", called["run_cmd"]+1))

    r = mod.VllmBuildRunner()
    r.run()

    assert called["run_cmd"] == 1
