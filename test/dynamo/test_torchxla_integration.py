# Owner(s): ["module: dynamo"]
import copy
import functools
import os
import unittest

import torch

has_torch_xla = True
try:
    import torch._dynamo.optimizations.torchxla_integration as integration
except ImportError:
    has_torch_xla = False

import torch.utils._pytree as pytree
from torch import fx, nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def forward(self, x, y):
        return x + y

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(10))


class MatmulModule(nn.Module):
    def __init__(self):
        super(MatmulModule, self).__init__()

    def forward(self, x, y):
        return x @ y

    def get_random_inputs(self):
        return (torch.randn(5, 100), torch.randn(100, 5))


class LinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(10),)


class ModuleInplaceUpdate(nn.Module):
    def __init__(self):
        super(ModuleInplaceUpdate, self).__init__()

    def forward(self, a, b):
        a.sub_(b)
        return b - 1, b + 1

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(10))


def allclose(expected, actual):
    def unwrap(cont):
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont

    expected = unwrap(expected)
    actual = unwrap(actual)

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(
            torch.allclose(a, b) for a, b in zip(expected, actual)
        )
    else:
        raise RuntimeError("Unexpected types")


@functools.lru_cache(None)
def should_run_torchxla_tests():
    """
    Run the tests if torch_xla is available and number of gpu devices is specified.
    """
    gpu_device_specified = int(os.environ.get("GPU_NUM_DEVICES", "0")) > 0
    return has_torch_xla and gpu_device_specified


def make_reuse_graph_test(module_class, niter=100):
    @unittest.skipIf(
        not should_run_torchxla_tests(),
        "Skip the tests since torch_xla is not available or XLA devices are not specified",
    )
    def test_wrapper(self):
        import torch_xla.core.xla_model as xm

        xla_dev = xm.xla_device()
        mod = module_class()
        xla_module = copy.deepcopy(mod).to(device=xla_dev)
        inputs = mod.get_random_inputs()
        optimized_mod = integration.extract_compiled_graph(
            fx.symbolic_trace(mod), inputs
        )

        for i in range(niter):
            rand_args = mod.get_random_inputs()
            orig_dev = rand_args[0].device
            rand_args_copy = copy.deepcopy(rand_args)

            # Can not simply call
            #   expected = mod(*rand_args)
            # Since we need use xla to calculate expected results
            xla_inputs = tuple(
                copy.deepcopy(inp).to(device=xla_dev) for inp in rand_args
            )
            xla_out = xla_module(*xla_inputs)
            # copy xla_inputs back to rand_args since the model may inplace update
            # the arguments
            rand_args = tuple(inp.to(device=orig_dev) for inp in xla_inputs)
            expected = pytree.tree_map(lambda o: o.to(device=orig_dev), xla_out)

            actual = optimized_mod(*rand_args_copy)

            if not allclose(expected, actual):
                print(
                    f"Incorrect results at iter {i}. expected\n{expected}, actual\n{actual}"
                )
                self.assertTrue(False)

            # make sure arguments match after calling the model forward method
            # to handle inplace updates.
            if not allclose(rand_args, rand_args_copy):
                print(
                    f"Incorrect updated arguments at iter {i}. expected\n{rand_args}, actual\n{rand_args_copy}"
                )
                self.assertTrue(False)

    return test_wrapper


class TorchXLAReuseGraphTest(unittest.TestCase):
    test_basic = make_reuse_graph_test(BasicModule)
    test_matmul = make_reuse_graph_test(MatmulModule)
    test_linear = make_reuse_graph_test(LinearModule)
    test_inplace_update = make_reuse_graph_test(ModuleInplaceUpdate)
