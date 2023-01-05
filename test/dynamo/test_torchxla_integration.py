# Owner(s): ["module: dynamo"]
import copy

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import aot_module_simplified, make_boxed_compiler
from torch._dynamo import disable

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
        return (torch.randn(2, 10),)


class MaxPoolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

    def get_random_inputs(self):
        return (torch.randn(2, 3, 10, 10),)


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


def training_compiler(gm, example_inputs):
    @make_boxed_compiler
    @disable
    def fw_compiler(graph, inputs, *args, **kwargs):
        # tracing time inputs are FakeTensors, we can not pass them
        # to extract_compiled_graph directly since we can not extract
        # xla tensor id from fake tensors. Call extract_compiled_graph
        # lazily and trigger that for the first call with non-fake tensors.
        compiled_graph = None

        def optimized_mod(*args):
            nonlocal compiled_graph
            if compiled_graph is None:
                compiled_graph = integration.extract_compiled_graph(graph, args)
            return compiled_graph(*args)

        return optimized_mod

    return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler)


def model_iter_fn_train(mod, inputs):
    outputs = mod(*inputs)
    loss = outputs.mean()
    loss.backward()

    param_list = list(mod.parameters())
    return [param.grad for param in param_list]


def make_training_test(model_cls):
    @maybe_skip_torchxla_test
    def test_wrapper(self):
        import torch_xla.core.xla_model as xm

        xla_dev = xm.xla_device()
        model = model_cls()
        inputs = model.get_random_inputs()

        model = model.to(device=xla_dev)
        inputs = tuple(inp.to(device=xla_dev) for inp in inputs)

        # do baseline
        baseline_model = copy.deepcopy(model)
        baseline_inputs = copy.deepcopy(inputs)
        expected_output = model_iter_fn_train(baseline_model, baseline_inputs)

        compiler = training_compiler
        optimize_ctx = torch._dynamo.optimize(compiler, nopython=False)
        optimized_model_iter_fn = optimize_ctx(model_iter_fn_train)

        actual_output = optimized_model_iter_fn(model, inputs)
        print(f"expected_output:\n{expected_output}\nactual_output:\n{actual_output}")
        assert allclose(expected_output, actual_output)

    return test_wrapper


class TorchXLAReuseGraphTest(torch._dynamo.test_case.TestCase):
    test_basic = make_reuse_graph_test(BasicModule)
    test_matmul = make_reuse_graph_test(MatmulModule)
    test_linear = make_reuse_graph_test(LinearModule)
    test_inplace_update = make_reuse_graph_test(ModuleInplaceUpdate)

    test_training_linear = make_training_test(LinearModule)
    test_training_maxpool = make_training_test(MaxPoolModule)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
