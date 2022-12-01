# Owner(s): ["module: dynamo"]
import copy
import unittest

import torch

try:
    from .test_torchxla_util import maybe_skip_torchxla_test
except ImportError:
    from test_torchxla_util import maybe_skip_torchxla_test

try:
    import torch._dynamo.optimizations.torchxla_integration as integration
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as metrics
except ImportError:
    # tests using torch_xla will be skipped. It's fine to ignore the
    # importing error here.
    pass

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


def make_reuse_graph_test(module_class, niter=100):
    @maybe_skip_torchxla_test
    def test_wrapper(self):
        xla_dev = xm.xla_device()
        xla_module = module_class().to(device=xla_dev)
        inputs = tuple(x.to(device=xla_dev) for x in xla_module.get_random_inputs())
        metrics.clear_counters()
        optimized_mod = integration.extract_compiled_graph(
            fx.symbolic_trace(xla_module), inputs
        )

        for i in range(niter):
            xla_inputs = tuple(
                inp.to(device=xla_dev) for inp in xla_module.get_random_inputs()
            )
            xla_inputs_copy = copy.deepcopy(xla_inputs)

            expected = xla_module(*xla_inputs)
            # make sure above lazy computation is executed.
            xm.mark_step()

            actual = optimized_mod(*xla_inputs_copy)

            if not allclose(expected, actual):
                print(
                    f"Incorrect results at iter {i}. expected\n{expected}, actual\n{actual}"
                )
                self.assertTrue(False)

            # make sure arguments match after calling the model forward method
            # to handle inplace updates.
            if not allclose(xla_inputs, xla_inputs_copy):
                print(
                    f"Incorrect updated arguments at iter {i}. expected\n{xla_inputs}, actual\n{xla_inputs_copy}"
                )
                self.assertTrue(False)

    return test_wrapper


class TorchXLAReuseGraphTest(unittest.TestCase):
    test_basic = make_reuse_graph_test(BasicModule)
    test_matmul = make_reuse_graph_test(MatmulModule)
    test_linear = make_reuse_graph_test(LinearModule)
    test_inplace_update = make_reuse_graph_test(ModuleInplaceUpdate)
