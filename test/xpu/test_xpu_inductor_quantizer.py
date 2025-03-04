# Owner(s): ["module: intel"]

# This files serves as supplementary tests for the cases in `test/inductor/test_mkldnn_pattern_matcher`
# This files tests the issue cases that shown only in XPU mode.
import contextlib

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_quantization import _generate_qdq_quantized_model
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)


@config.patch({"freezing": True})
@config.patch({"force_disable_caches": True})
class TestXPUInductorQuantizer(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_check_fn,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        is_qat=False,
        dtype=None,
        is_dynamic=False,
        quantizer=None,
        compile_options={},  # noqa: B006
    ):
        counters.clear()
        torch._dynamo.reset()
        device_type = "xpu"
        if check_autocast == torch.bfloat16:
            maybe_autocast = torch.amp.autocast(
                device_type=device_type, dtype=torch.bfloat16
            )
            atol, rtol = 1e-2, 1e-2
        elif check_autocast == torch.float16:
            maybe_autocast = torch.amp.autocast(
                device_type=device_type, dtype=torch.float16
            )
            atol, rtol = 1e-2, 1e-2
        else:
            assert check_autocast == torch.float32
            maybe_autocast = contextlib.nullcontext()
        convert_model = _generate_qdq_quantized_model(
            mod, inputs, is_qat, is_dynamic, quantizer
        )
        with torch.no_grad(), maybe_autocast:
            compiled_model = torch.compile(convert_model)
            ref = compiled_model(*self._clone_inputs(inputs))
            res = mod(*self._clone_inputs(inputs))
            relative_err = torch.mean(torch.abs(res - ref) / ref.abs().clamp(1e-6))
            self.assertTrue(relative_err < 0.1)
            matcher_check_fn()

    def test_qlinear_pointwise_binary_3d(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                orig = x
                out = self.linear(x)
                return out + orig

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
            )
            self.assertEqual(counters["inductor"]["qlinear_binary_matcher_count"], 1)

        mod = Model().xpu()
        inputs = (torch.rand(2, 3, 10, device="xpu"),)
        self._test_common(mod, inputs, matcher_check_fn)


instantiate_parametrized_tests(TestXPUInductorQuantizer)

if __name__ == "__main__":
    run_tests()
