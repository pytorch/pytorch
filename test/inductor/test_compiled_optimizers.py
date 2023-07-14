# Owner(s): ["module: inductor"]

import sys
import unittest

from copy import deepcopy

import torch

import torch._inductor

from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda, requires_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda, requires_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


def make_test(optim_cls, closure=None, **kwargs):
    @requires_cuda()
    def test_fn(self):
        input = torch.ones([10, 10], device="cuda:0")
        model_eager = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device="cuda:0") for _ in range(2)]
        )
        model_eager(input).sum().backward()

        input = torch.ones([10, 10], device="cuda:0")
        model_compiled = deepcopy(model_eager)
        model_compiled(input).sum().backward()

        opt_eager = optim_cls(model_eager.parameters(), **kwargs)
        opt_compiled = optim_cls(model_compiled.parameters(), **kwargs)
        # run the patcher so that step has the expected structure
        torch._dynamo.eval_frame.TorchPatcher.patch()

        # unwrap step to avoid a deliberate graph break due to
        # a limitation of functionalization/no_grad detection
        # see the [Note on graph break] in optimizer.py
        # This ignores the outer _use_grad_if_differentiable wrapper
        # and instead manually disables grad before calling step, which is fine
        # for now as dynamo does not support differentiable optimizers anyway
        step_fn = opt_compiled.step.__wrapped__
        if closure is not None:

            def fn():
                step_fn(opt_compiled, closure)

        else:

            def fn():
                step_fn(opt_compiled)

        with torch.set_grad_enabled(False):
            torch.compile(fn, backend="inductor", fullgraph=True)()
            opt_eager.step()

        self.assertEqual(
            list(model_eager.parameters()), list(model_compiled.parameters())
        )
        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    return test_fn


class CompiledOptimizerTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    test_adam = make_test(torch.optim.Adam, lr=0.01)
    test_adam_weight_decay = make_test(torch.optim.Adam, lr=0.01, weight_decay=0.01)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
