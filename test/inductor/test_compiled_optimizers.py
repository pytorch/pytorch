# Owner(s): ["module: inductor"]

import sys
import unittest

from copy import deepcopy

import torch

import torch._inductor

# The rest of the optimizers not yet imported: Adamax, ASGD, LBFGS, NAdam, RAdam, SGD, SparseAdam
from torch.optim import Adadelta, Adagrad, Adam, AdamW, RMSprop, Rprop

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


def compile_opt(opt_compiled, closure=None):
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

    return torch.compile(fn, backend="inductor", fullgraph=True)


def make_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    @requires_cuda()
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
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
        compiled_step = compile_opt(opt_compiled, closure=closure)

        with torch.set_grad_enabled(False):
            compiled_step()
            opt_eager.step()

        self.assertEqual(
            list(model_eager.parameters()), list(model_compiled.parameters())
        )

        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, kernel_count
            )

    return test_fn


def make_recompile_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    @requires_cuda()
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device="cuda:0")
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device="cuda:0") for _ in range(2)]
        )
        model(input).sum().backward()

        opt_compiled = optim_cls(model.parameters(), **kwargs)
        compiled_step = compile_opt(opt_compiled)

        # check no recompile here
        with torch.set_grad_enabled(False):
            compiled_step()

            compiled_step()

            # perturb state to force recompile
            # Adagrad doesn't reinitialize state on each step
            if optim_cls is Adagrad:
                opt_compiled.param_groups[0]["lr"] = 0.02
            else:
                opt_compiled.state.clear()

            compiled_step()

        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            # multiply by 2 to account for the recompile
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, 2 * kernel_count
            )

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

    test_adam = make_test(Adam, lr=0.01)
    test_adam_weight_decay = make_test(Adam, lr=0.01, weight_decay=0.01)
    test_adam_amsgrad = make_test(Adam, lr=0.01, amsgrad=True)
    test_adam_maximize = make_test(Adam, lr=0.01, maximize=True)
    test_adam_weight_decay_and_maximize = make_test(
        Adam, lr=0.01, weight_decay=0.01, maximize=True
    )
    test_adam_everything = make_test(
        Adam, lr=0.01, weight_decay=1.0, amsgrad=True, capturable=True, maximize=True
    )

    test_adamw = make_test(AdamW, lr=0.01)
    # Need to an impl which does not use python scalars
    # test_adamax = make_test(Adamax, lr=0.01)
    # test_nadam = make_test(NAdam, lr=0.01)
    test_rprop = make_test(Rprop, kernel_count=1, lr=0.01)
    test_rmsprop = make_test(RMSprop, kernel_count=1, lr=0.01)
    test_adadelta = make_test(Adadelta, kernel_count=1, lr=0.01)
    test_adagrad = make_test(Adagrad, kernel_count=5, lr=0.01)
    # test_sgd = make_test(SGD, kernel_count=1, lr=0.01)

    test_adam_recompile = make_recompile_test(Adam, lr=0.01)
    test_adamw_recompile = make_recompile_test(AdamW, lr=0.01)
    # Need an impl which does not use python scalars
    # test_adamax_recompile = make_recompile_test(Adamax, lr=0.01)
    # test_nadam_recompile = make_recompile_test(NAdam, lr=0.01)
    test_rprop_recompile = make_recompile_test(Rprop, kernel_count=1, lr=0.01)
    test_rmsprop_recompile = make_recompile_test(RMSprop, kernel_count=1, lr=0.01)
    test_adadelta_recompile = make_recompile_test(Adadelta, kernel_count=1, lr=0.01)
    test_adagrad_recompile = make_recompile_test(Adagrad, kernel_count=5, lr=0.01)
    # test_sgd_recompile = make_recompile_test(SGD, kernel_count=1, lr=0.01)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
