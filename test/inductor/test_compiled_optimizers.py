# Owner(s): ["module: inductor"]

import sys
import unittest
import weakref

from copy import deepcopy
from typing import NamedTuple

import torch

import torch._inductor

# The rest of the optimizers not yet imported: Adamax, LBFGS, RAdam, SGD, SparseAdam
from torch.optim import Adadelta, Adagrad, Adam, AdamW, ASGD, NAdam, RMSprop, Rprop

from torch.testing._internal.common_optimizers import optim_db

from torch.testing._internal.common_utils import TestCase

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


class KernelCounts(NamedTuple):
    multitensor: int
    singletensor: int


# With different settings for certain
# tests you can get different kernel counts
# This maps the test name to the
# expected kernel count
KERNEL_COUNT_OVERRIDES = {
    "test_rmsprop_foreach_weight_decay_cpu": 12,
    "test_nadam_foreach_weight_decay_momentum_decay_cpu": 20,
    "test_adamw_foreach_amsgrad_capturable_cuda": 3,
    "test_adamw_amsgrad_capturable_cuda": 6,
    "test_adam_amsgrad_capturable_cuda": 6,
    "test_adadelta_foreach_weight_decay_maximize_cpu": 12,
    "test_adadelta_foreach_rho_weight_decay_cpu": 12,
    "test_adadelta_foreach_weight_decay_cpu": 12,
}

# also tracks currently supported optimizers
KERNEL_COUNTS = {
    Adam: KernelCounts(multitensor=2, singletensor=8),
    AdamW: KernelCounts(multitensor=2, singletensor=8),
    NAdam: KernelCounts(multitensor=2, singletensor=12),
    Rprop: KernelCounts(multitensor=1, singletensor=4),
    RMSprop: KernelCounts(multitensor=1, singletensor=4),
    Adadelta: KernelCounts(multitensor=1, singletensor=4),
    Adagrad: KernelCounts(multitensor=5, singletensor=8),
    ASGD: KernelCounts(multitensor=2, singletensor=12),
}


def build_compiled_opt_kwarg_db():
    compiled_opt_db = []
    for optim_info in optim_db:
        if optim_info.optim_cls not in KERNEL_COUNTS:
            continue

        for optim_inputs in optim_info.optim_inputs_func():
            for device in ["cpu", "cuda"]:
                for foreach in [True, False]:
                    if device == "cpu" and "capturable" in optim_inputs.kwargs:
                        continue

                    kwargs = dict(optim_inputs.kwargs)
                    name = (
                        f"test_{optim_info.optim_cls.__name__.lower()}"
                        f"{'_foreach' if foreach else ''}"
                    )

                    for key in optim_inputs.kwargs:
                        if key == "lr":
                            continue
                        name += "_" + key

                    name += f"_{device}"

                    # Eager for-loop impl doesn't support capturable ASGD
                    if name == "test_asgd_capturable_cuda":
                        continue

                    kwargs["foreach"] = foreach
                    kwargs["device"] = device
                    if name in KERNEL_COUNT_OVERRIDES:
                        kwargs["kernel_count"] = KERNEL_COUNT_OVERRIDES[name]
                    else:
                        kwargs["kernel_count"] = (
                            KERNEL_COUNTS[optim_info.optim_cls].multitensor
                            if foreach and device == "cuda"
                            else KERNEL_COUNTS[optim_info.optim_cls].singletensor
                        )

                    # Note on tolerances:
                    # test_adadelta_foreach_rho_weight_decay_cuda
                    # Mismatched elements: 1 / 100 (1.0%)
                    # Greatest absolute difference: 2.0936131477355957e-05 at index (2, 7) (up to 2e-05 allowed)
                    # Greatest relative difference: 8.520411211065948e-05 at index (2, 7) (up to 1e-06 allowed)
                    if optim_info.optim_cls is Adadelta:
                        kwargs["rtol"] = 2e-5
                        kwargs["atol"] = 2e-5

                    compiled_opt_db.append((optim_info.optim_cls, name, kwargs))

    return compiled_opt_db


COMPILED_OPT_KWARG_DB = build_compiled_opt_kwarg_db()

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

    # unwrap step TWICE to avoid a deliberate graph break due to
    # a limitation of functionalization/no_grad detection
    # see the [Note on graph break] in optimizer.py
    # This ignores the outer _use_grad_if_differentiable wrapper
    # and instead manually disables grad before calling step, which is fine
    # for now as dynamo does not support differentiable optimizers anyway
    step_fn = opt_compiled.step.__wrapped__.__wrapped__
    if closure is not None:

        def fn():
            step_fn(opt_compiled, closure)

    else:

        def fn():
            step_fn(opt_compiled)

    return torch.compile(fn, backend="inductor", fullgraph=True)


def make_test(
    optim_cls,
    closure=None,
    kernel_count=2,
    device="cuda",
    atol=None,
    rtol=None,
    **kwargs,
):
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device=device)
        model_eager = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=device) for _ in range(2)]
        )
        model_eager(input).sum().backward()

        input = torch.ones([10, 10], device=device)
        model_compiled = deepcopy(model_eager)
        model_compiled(input).sum().backward()

        opt_eager = optim_cls(model_eager.parameters(), **kwargs)
        opt_compiled = optim_cls(model_compiled.parameters(), **kwargs)
        compiled_step = compile_opt(opt_compiled, closure=closure)

        with torch.set_grad_enabled(False):
            compiled_step()
            compiled_step()
            opt_eager.step()
            opt_eager.step()

        self.assertEqual(
            list(model_eager.parameters()),
            list(model_compiled.parameters()),
            atol=atol,
            rtol=rtol,
        )

        # currently we don't mutate step properly until we resolve
        # https://github.com/pytorch/pytorch/issues/115679
        if optim_cls not in (Adadelta, Rprop, RMSprop):
            for p_eager, p_compiled in zip(
                model_eager.parameters(), model_compiled.parameters()
            ):
                self.assertEqual(
                    opt_eager.state[p_eager],
                    opt_compiled.state[p_compiled],
                    atol=atol,
                    rtol=rtol,
                )

        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, kernel_count
            )

    if device == "cuda":
        test_fn = requires_cuda()(test_fn)

    return test_fn


def make_recompile_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    @requires_cuda()
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device="cuda")
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device="cuda") for _ in range(2)]
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

    # test_sgd = make_test(SGD, kernel_count=1, lr=0.01)

    test_adam_recompile = make_recompile_test(Adam, lr=0.01)
    test_adamw_recompile = make_recompile_test(AdamW, lr=0.01)
    # Need an impl which does not use python scalars
    # test_adamax_recompile = make_recompile_test(Adamax, lr=0.01)
    test_nadam_recompile = make_recompile_test(NAdam, lr=0.01)
    test_rprop_recompile = make_recompile_test(Rprop, kernel_count=1, lr=0.01)
    test_rmsprop_recompile = make_recompile_test(RMSprop, kernel_count=1, lr=0.01)
    test_adadelta_recompile = make_recompile_test(Adadelta, kernel_count=1, lr=0.01)
    test_adagrad_recompile = make_recompile_test(Adagrad, kernel_count=5, lr=0.01)
    test_asgd_recompile_default = make_recompile_test(ASGD, kernel_count=2, lr=0.01)
    test_asgd_recompile_single = make_recompile_test(
        ASGD, kernel_count=12, lr=0.01, foreach=False
    )
    test_asgd_recompile_foreach = make_recompile_test(
        ASGD, kernel_count=2, lr=0.01, foreach=True
    )
    # test_sgd_recompile = make_recompile_test(SGD, kernel_count=1, lr=0.01)

    @requires_cuda()
    def test_static_address_finalizer(self):
        import gc

        gc.disable()
        p_ref = None

        def fn():
            nonlocal p_ref
            mod = torch.nn.Linear(10, 10, device="cuda:0", bias=False)
            for p in mod.parameters():
                p.grad = torch.rand_like(p)

            opt = torch.optim.Adam(mod.parameters(), lr=0.1)

            def fn():
                opt.step()

            with torch.set_grad_enabled(False):
                step_fn_compiled = torch.compile(fn)
                step_fn_compiled()
            p_ref = weakref.ref(p)
            self.assertTrue(p_ref() is not None)

        fn()

        self.assertTrue(p_ref() is None)
        gc.enable()


for optim_cls, name, kwargs in COMPILED_OPT_KWARG_DB:
    setattr(CompiledOptimizerTests, name, make_test(optim_cls, **kwargs))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
