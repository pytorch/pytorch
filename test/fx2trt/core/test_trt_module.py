import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt import (
    TRTInterpreter,
    InputTensorSpec,
    TRTModule,
)
from torch.testing._internal.common_utils import TestCase, run_tests
import os


class TestTRTModule(TestCase):
    def test_save_and_load_trt_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(mod, input_specs=InputTensorSpec.from_tensors(inputs))
        trt_mod = TRTModule(*interp.run(fp16_mode=False))
        torch.save(trt_mod, "trt.pt")
        reload_trt_mod = torch.load("trt.pt")

        torch.testing.assert_allclose(
            reload_trt_mod(inputs[0].cuda()).cpu(), ref_output, rtol=1e-04, atol=1e-04
        )
        os.remove(f"{os.getcwd()}/trt.pt")

    def test_save_and_load_state_dict(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(mod, input_specs=InputTensorSpec.from_tensors(inputs))
        trt_mod = TRTModule(*interp.run(fp16_mode=False))
        st = trt_mod.state_dict()

        new_trt_mod = TRTModule()
        new_trt_mod.load_state_dict(st)

        torch.testing.assert_allclose(
            new_trt_mod(inputs[0].cuda()).cpu(), ref_output, rtol=1e-04, atol=1e-04
        )

if __name__ == '__main__':
    run_tests()
