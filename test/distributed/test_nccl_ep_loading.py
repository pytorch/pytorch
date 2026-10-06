# Owner(s): ["oncall: distributed"]

import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributed._token_switch import _import_nccl_ep
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestNcclEpLoading(TestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.torch_dir = self.root / "installed" / "torch"
        self.home = self.torch_dir / "share" / "nccl_ep"
        headers = (
            "nccl_ep.h",
            "nccl_ep/common.hpp",
            "nccl_ep/ep_enums.h",
            "nccl_ep/device/ht_ep.cuh",
            "nccl.h",
            "nccl_device.h",
            "nccl_device/impl/core__types.h",
        )
        for relative in headers:
            header = self.home / "include" / relative
            header.parent.mkdir(parents=True, exist_ok=True)
            header.touch()
        self.ep = SimpleNamespace(__file__=str(self.torch_dir / "_nccl_ep.so"))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.import_module = stack.enter_context(
            mock.patch("torch.distributed._token_switch.importlib")
        ).import_module
        self.import_module.return_value = self.ep
        self.which = stack.enter_context(
            mock.patch(
                "torch.distributed._token_switch.shutil.which", return_value=None
            )
        )

    def test_load_without_nccl_python_packages(self):
        self.assertIs(_import_nccl_ep(), self.ep)
        self.import_module.assert_called_once_with("torch._nccl_ep")
        self.assertEqual(os.environ["NCCL_EP_HOME"], str(self.home))
        self.assertEqual(os.environ["NCCL_HOME"], str(self.home))

    def test_relocated_editable_install(self):
        relocated = self.root / "relocated" / "torch"
        relocated.parent.mkdir()
        self.torch_dir.rename(relocated)
        self.ep.__file__ = str(relocated / "_nccl_ep.so")
        source = self.root / "source" / "torch" / "__init__.py"
        with mock.patch.object(torch, "__file__", str(source)):
            _import_nccl_ep()
        home = str(relocated / "share" / "nccl_ep")
        self.assertEqual(os.environ["NCCL_EP_HOME"], home)
        self.assertEqual(os.environ["NCCL_HOME"], home)

    def test_explicit_homes(self):
        ep_home = self.root / "ep"
        nccl_home = self.root / "nccl"
        shutil.copytree(self.home, ep_home)
        shutil.copytree(self.home, nccl_home)
        shutil.rmtree(self.home)
        overrides = {"NCCL_EP_HOME": str(ep_home), "NCCL_HOME": str(nccl_home)}
        os.environ.update(overrides)
        _import_nccl_ep()
        self.assertEqual(dict(os.environ), overrides)

    def test_jit_overrides_take_precedence(self):
        custom = self.root / "custom"
        self.home.rename(custom)
        overrides = {
            "NCCL_EP_HOME": "/unused/ep",
            "NCCL_HOME": "/unused/nccl",
            "NCCL_EP_JIT_SOURCE_DIR": str(custom / "include" / "nccl_ep"),
            "NCCL_EP_JIT_BUILD_INCLUDE_DIR": str(custom / "include"),
        }
        os.environ.update(overrides)
        _import_nccl_ep()
        self.assertEqual(dict(os.environ), overrides)

    @parametrize("name", ["NCCL_EP_HOME", "NCCL_HOME"])
    def test_invalid_override(self, name):
        os.environ[name] = str(self.root / "missing")
        with self.assertRaisesRegex(ImportError, "runtime JIT header is missing"):
            _import_nccl_ep()
        self.assertEqual(os.environ[name], str(self.root / "missing"))

    def test_empty_overrides_use_packaged_headers(self):
        os.environ.update(
            NCCL_EP_HOME="",
            NCCL_HOME="",
            NCCL_EP_JIT_SOURCE_DIR="",
            NCCL_EP_JIT_BUILD_INCLUDE_DIR="",
        )
        _import_nccl_ep()
        self.assertEqual(os.environ["NCCL_EP_HOME"], str(self.home))
        self.assertEqual(os.environ["NCCL_HOME"], str(self.home))

    @parametrize(
        "header",
        ["nccl_ep.h", "nccl_ep/device/ht_ep.cuh", "nccl_device/impl/core__types.h"],
    )
    def test_missing_packaged_header(self, header):
        (self.home / "include" / header).unlink()
        with self.assertRaisesRegex(ImportError, "runtime JIT header is missing") as cm:
            _import_nccl_ep()
        self.assertIn(header, str(cm.exception))
        self.assertEqual(dict(os.environ), {})

    def test_extension_not_built(self):
        error = ModuleNotFoundError("no EP extension", name="torch._nccl_ep")
        self.import_module.side_effect = error
        with self.assertRaisesRegex(ImportError, "built with USE_NCCL_EP=1") as cm:
            _import_nccl_ep()
        self.assertIs(cm.exception.__cause__, error)
        self.assertEqual(dict(os.environ), {})

    @parametrize(
        "error",
        [
            ImportError("libnccl_ep.so.0: cannot open shared object file"),
            ModuleNotFoundError("missing native dependency", name="dependency"),
        ],
    )
    def test_native_load_error(self, error):
        self.import_module.side_effect = error
        with self.assertRaisesRegex(ImportError, "native dependencies") as cm:
            _import_nccl_ep()
        self.assertIn(str(error), str(cm.exception))
        self.assertIs(cm.exception.__cause__, error)
        self.assertEqual(dict(os.environ), {})

    @parametrize("selector", [None, "NVCC", "NCCL_EP_JIT_NVCC"])
    def test_runtime_cuda_headers(self, selector):
        toolkit = self.root / "cuda"
        (toolkit / "include").mkdir(parents=True)
        nvcc = str(toolkit / "bin" / "nvcc")
        if selector:
            os.environ[selector] = nvcc
        self.which.return_value = nvcc
        _import_nccl_ep()
        self.assertEqual(
            os.environ["NCCL_EP_JIT_CUDA_INCLUDE_DIR"], str(toolkit / "include")
        )
        self.which.assert_called_once_with(nvcc if selector else "nvcc")

    @parametrize("name", ["CUDA_HOME", "CUDA_PATH", "NCCL_EP_JIT_CUDA_INCLUDE_DIR"])
    def test_preserve_cuda_override(self, name):
        os.environ[name] = "/custom/cuda"
        _import_nccl_ep()
        self.assertEqual(os.environ[name], "/custom/cuda")
        self.which.assert_not_called()


if __name__ == "__main__":
    run_tests()
