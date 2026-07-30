# Owner(s): ["module: cuda"]
import builtins
import importlib.util
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


_PCG_NAMES = ["CUDAGraphSequence", "piecewise_graph", "no_graph", "force_no_graph"]


def _package_available() -> bool:
    return importlib.util.find_spec("piecewise_cuda_graphs") is not None


class TestPiecewiseCudaGraphForwarding(TestCase):
    def test_names_in_all(self):
        for name in _PCG_NAMES:
            self.assertIn(name, torch.cuda.__all__)

    def test_forwards_when_present(self):
        if not _package_available():
            self.skipTest("piecewise_cuda_graphs not installed")
        import piecewise_cuda_graphs

        for name in _PCG_NAMES:
            self.assertIs(
                getattr(torch.cuda, name), getattr(piecewise_cuda_graphs, name)
            )

    def test_raises_when_absent(self):
        # Simulate the package being uninstalled: the name still resolves, but
        # calling/instantiating it raises an ImportError with an install hint.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.split(".")[0] == "piecewise_cuda_graphs":
                raise ImportError("simulated missing package")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            for name in _PCG_NAMES:
                placeholder = getattr(torch.cuda, name)
                with self.assertRaisesRegex(ImportError, "torchannex"):
                    placeholder()


if __name__ == "__main__":
    run_tests()
