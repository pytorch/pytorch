# Owner(s): ["module: cpp"]

import sysconfig
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    run_tests,
    TestCase,
)


def _torch_version_less_than(major, minor):
    parts = torch.__version__.split(".")
    cur_major = int(parts[0])
    cur_minor = int(parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0])
    return (cur_major < major) or (cur_major == major and cur_minor < minor)


# These shims live in libtorch_python and require holding the GIL, so unlike the
# libtorch_agn_* extensions the test extension is not libtorch-only. It is still
# abi3 (py_limited_api); it is skipped on free-threaded (GIL-disabled) builds.
@unittest.skipIf(
    sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    "python-interop stable shims require holding the GIL",
)
@unittest.skipIf(
    _torch_version_less_than(2, 14),
    "python-interop stable shims require PyTorch >= 2.14",
)
class TestLibtorchPythonInterop(TestCase):
    """Tests for the python-aware stable shims (torch/csrc/stable/python).

    These depend on both libtorch and libtorch_python, so they are exercised by
    a dedicated extension (libtorch_python_interop_2_14) that links torch_python and
    exposes a PyMethodDef module function -- kept separate from the libtorch-only
    libtorch_agn_* family and its tests.
    """

    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).parent
        try:
            import libtorch_python_interop_2_14  # noqa: F401
        except Exception:
            install_cpp_extension(
                extension_root=base_dir / "libtorch_python_interop_2_14_extension"
            )

    @onlyCPU
    def test_pyobject_roundtrip_shares_storage(self, device):
        import libtorch_python_interop_2_14 as ext

        x = torch.randn(3, 4)
        y = ext.pyobject_roundtrip(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y, x)
        # from_pyobject / to_pyobject share the underlying TensorImpl.
        self.assertEqual(y.data_ptr(), x.data_ptr())
        x.add_(1)
        self.assertEqual(y, x)

    @onlyCPU
    def test_pyobject_to_parameter_type(self, device):
        import libtorch_python_interop_2_14 as ext

        x = torch.randn(2, 2)
        p = ext.pyobject_to_type(x, torch.nn.Parameter)
        self.assertIsInstance(p, torch.nn.Parameter)
        self.assertEqual(p.detach(), x)

    @onlyCPU
    def test_pyobject_sum(self, device):
        import libtorch_python_interop_2_14 as ext

        # Exercises real work on the tensor inside C++ (torch::stable::sum)
        # via a tensor obtained from from_pyobject.
        x = torch.randn(3, 4)
        s = ext.pyobject_sum(x)
        self.assertIsInstance(s, torch.Tensor)
        self.assertEqual(s, x.sum())

    @onlyCPU
    def test_pyobject_roundtrip_non_tensor_raises(self, device):
        import libtorch_python_interop_2_14 as ext

        with self.assertRaises(Exception):
            ext.pyobject_roundtrip("not a tensor")


instantiate_device_type_tests(TestLibtorchPythonInterop, globals(), except_for=None)

if __name__ == "__main__":
    run_tests()
