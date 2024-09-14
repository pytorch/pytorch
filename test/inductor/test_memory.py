# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.inductor_utils import HAS_GPU


class TestOperatorReorderForPeakMemory(TestCase):
    """
    The test case is for the Foo() model defined below.

    The default compiled graph is
    graph():
        ...
        %op0 : [num_users=2] = call_function[...](args = (%primals_2, %primals_1), ...)
        %op1 : [num_users=2] = call_function[...](args = (%primals_2, %primals_3), ...)
        %op2 : [num_users=1] = call_function[...](args = (%op0, %primals_4), ...)
        %op3 : [num_users=1] = call_function[...](args = (%op1, %primals_5), ...)
        %op4 : [num_users=1] = call_function[...](args = (%op2,), ...)
        %op5 : [num_users=1] = call_function[...](args = (%op3,), ...)
        %op6_op7 : [num_users=1] = call_function[...](args = (%op5, %op4), ...)

    With reordering, the compiled graph is
    graph():
        ...
        %op1 : [num_users=2] = call_function[...](args = (%primals_2, %primals_3), ...)
        %op3 : [num_users=1] = call_function[...](args = (%op1, %primals_5), ...)
        %op5 : [num_users=1] = call_function[...](args = (%op3,), ...)
        %op0 : [num_users=2] = call_function[...](args = (%primals_2, %primals_1), ...)
        %op2 : [num_users=1] = call_function[...](args = (%op0, %primals_4), ...)
        %op4 : [num_users=1] = call_function[...](args = (%op2,), ...)
        %op6_op7 : [num_users=1] = call_function[...](args = (%op5, %op4), ...)
    """

    @patch.object(torch._inductor.config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Parameter(torch.ones(1, 10))
                self.w2 = torch.nn.Parameter(torch.ones(1, 1))
                self.w3 = torch.nn.Parameter(torch.ones(10, 1))
                self.w4 = torch.nn.Parameter(torch.ones(1, 10))

            def forward(self, x):
                t1 = torch.matmul(x, self.w1)
                t2 = torch.matmul(x, self.w2)
                t3 = torch.matmul(t1, self.w3)
                t4 = torch.matmul(t2, self.w4)
                return t3.sum() + t4.sum()

        dev = "cuda"  # torch.cuda.current_device()
        inp = torch.ones((2048, 1), device=dev)
        model = Foo().to(dev)
        outp_corr = model(inp)
        compiled_model = torch.compile(model)
        code = run_and_get_triton_code(compiled_model, inp)
        (
            FileCheck()
            .check("def call(args):")
            .check("buf1 = ")
            .check("buf3 = ")
            .check("buf5 = ")
            .check("buf0 = ")
            .check("buf2 = ")
            .check("buf4 = ")
            .check("buf7 = ")
            .run(code)
        )
        # check for correctness
        outp = compiled_model(inp)
        self.assertTrue(same(outp, outp_corr))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
