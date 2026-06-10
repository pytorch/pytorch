# Owner(s): ["module: dynamo"]

import re

import torch
import torch._dynamo.test_case
from torch._dynamo.convert_frame import fullgraph_capture
from torch._dynamo.utils import get_metrics_context
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo


class SimpleLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


@torch._dynamo.config.patch(generate_pycode=True)
@skipIfTorchDynamo("Not suitable for generate_pycode=True")
class TestPycode(torch._dynamo.test_case.TestCase):
    def test_pycode_module(self):
        mod = SimpleLinearModule()
        x = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(mod, (x,), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = self._modules['linear']._parameters['weight']
__arg1 = self._modules['linear']._parameters['bias']
__arg2 = x
__graph_out = __compiled_fn_<ID>(__arg0, __arg1, __arg2)
__stack0 = __graph_out[0]
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(mod, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    def test_pycode_dict_output(self):
        def fn(x, y):
            return {"sum": x + y, "diff": x - y, "prod": x * y}

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(fn, (x, y), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = x
__arg1 = y
__graph_out = __compiled_fn_<ID>(__arg0, __arg1)
__stack0 = {'sum': __graph_out[0], 'diff': __graph_out[1], 'prod': __graph_out[2]}
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_pycode_wrapper_source(self):
        mod = SimpleLinearModule()
        x = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(mod, (x,), {})
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        source = runtime_env._build_python_wrapper_source()
        source = re.sub(r"__compiled_fn_\d+_[0-9a-f_]+", "__compiled_fn_<ID>", source)
        self.assertExpectedInline(
            source,
            """\
def __generate_func__():
    def __dynamo_func__(self, x):
        __arg0 = self._modules['linear']._parameters['weight']
        __arg1 = self._modules['linear']._parameters['bias']
        __arg2 = x
        __graph_out = __compiled_fn_<ID>(__arg0, __arg1, __arg2)
        __stack0 = __graph_out[0]
        __ret = __stack0
        return __ret
    return __dynamo_func__
__dynamo_generated_func__ = __generate_func__()""",
        )

    def test_pycode_default_args(self):
        def fn(x, y, scale=1.0, bias=0.0):
            return x * scale + y + bias

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(fn, (x, y), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = x
__arg1 = y
__graph_out = __compiled_fn_<ID>(__arg0, __arg1)
__stack0 = __graph_out[0]
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
