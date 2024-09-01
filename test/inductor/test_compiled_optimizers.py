# Owner(s): ["module: inductor"]

import sys
import unittest
import weakref
from contextlib import ExitStack
from copy import deepcopy
from typing import NamedTuple

import torch
import torch._inductor
import torch._inductor.cudagraph_trees
import torch.optim.lr_scheduler
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
    skipXPUIf,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    has_triton,
)
from torch.testing._internal.triton_utils import requires_cuda, requires_gpu


# Note: we use atypical values to amplify error
LR_SCHEDULER_TO_KWARGS = {
    LambdaLR: {"lr_lambda": lambda x: 10},
    MultiplicativeLR: {"lr_lambda": lambda x: 10},
    StepLR: {"step_size": 1, "gamma": 100},
    MultiStepLR: {"milestones": [1, 2], "gamma": 100},
    ExponentialLR: {"gamma": 100},
    CosineAnnealingLR: {"T_max": 7},
    # These schedulers have memory leaks in eager
    # https://github.com/pytorch/pytorch/issues/126131
    # SequentialLR: {"schedulers": None, "milestones": [1, 2]},
    # ChainedScheduler: {"schedulers": None},
    CyclicLR: {"base_lr": 0.001, "max_lr": 0.02, "cycle_momentum": False},
    CosineAnnealingWarmRestarts: {"T_0": 1},
    OneCycleLR: {
        "max_lr": 0.02,
        "cycle_momentum": False,
        "steps_per_epoch": 1,
        "epochs": 10,
    },
    ConstantLR: {"factor": 0.001},
    LinearLR: {},
    ReduceLROnPlateau: {"factor": 0.99, "patience": 1},
    PolynomialLR: {},
}


def create_scheduler(scheduler, optim):
    kwargs = LR_SCHEDULER_TO_KWARGS[scheduler]
    if "schedulers" in kwargs:
        kwargs["schedulers"] = [
            create_scheduler(torch.optim.lr_scheduler.ConstantLR, optim)
            for _ in range(2)
        ] + [create_scheduler(torch.optim.lr_scheduler.LambdaLR, optim)]

    if scheduler == ChainedScheduler:
        return scheduler(**kwargs)
    else:
        return scheduler(optim, **kwargs)


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
    "test_adamw_amsgrad_capturable_foreach_cuda": 3,
    "test_adamw_amsgrad_capturable_foreach_xpu": 3,
    "test_adamw_amsgrad_capturable_cuda": 6,
    "test_adamw_amsgrad_capturable_xpu": 6,
    "test_adamw_tensor_lr_amsgrad_capturable_foreach_cuda": 3,
    "test_adamw_tensor_lr_amsgrad_capturable_foreach_xpu": 3,
    "test_adamw_tensor_lr_amsgrad_capturable_cuda": 6,
    "test_adamw_tensor_lr_amsgrad_capturable_xpu": 6,
    "test_adam_tensor_lr_amsgrad_capturable_cuda": 6,
    "test_adam_tensor_lr_amsgrad_capturable_xpu": 6,
    "test_adam_amsgrad_capturable_cuda": 6,
    "test_adam_amsgrad_capturable_xpu": 6,
    "test_adadelta_tensor_lr_capturable_cuda": 6,
    "test_adadelta_tensor_lr_capturable_xpu": 6,
    "test_rmsprop_tensor_lr_capturable_cuda": 6,
    "test_rmsprop_tensor_lr_capturable_xpu": 6,
    "test_adadelta_tensor_lr_capturable_foreach_cuda": 4,
    "test_adadelta_tensor_lr_capturable_foreach_xpu": 4,
    "test_adadelta_foreach_weight_decay_maximize_cpu": 12,
    "test_adadelta_foreach_rho_weight_decay_cpu": 12,
    "test_adadelta_foreach_weight_decay_cpu": 12,
    "test_sgd_foreach_momentum_weight_decay_cpu": 16,
    "test_sgd_foreach_momentum_nesterov_weight_decay_cpu": 16,
    "test_sgd_momentum_dampening_foreach_cuda": 5,
    "test_sgd_momentum_dampening_foreach_xpu": 5,
    "test_sgd_momentum_foreach_cuda": 5,
    "test_sgd_momentum_foreach_xpu": 5,
    "test_sgd_weight_decay_maximize_cuda": 4,
    "test_sgd_weight_decay_maximize_xpu": 4,
    "test_sgd_weight_decay_maximize_cpu": 4,
    "test_sgd_weight_decay_cpu": 4,
    "test_sgd_weight_decay_cuda": 4,
    "test_sgd_weight_decay_xpu": 4,
    "test_sgd_momentum_weight_decay_foreach_cuda": 2,
    "test_sgd_momentum_weight_decay_foreach_xpu": 2,
    "test_sgd_momentum_nesterov_weight_decay_foreach_cuda": 2,
    "test_sgd_momentum_nesterov_weight_decay_foreach_xpu": 2,
    "test_sgd_cuda": 4,
    "test_sgd_cpu": 4,
    "test_sgd_xpu": 4,
    "test_rmsprop_tensor_lr_capturable_foreach_cuda": 4,
    "test_rmsprop_tensor_lr_capturable_foreach_xpu": 4,
    "test_adagrad_initial_accumulator_value_weight_decay_foreach_xpu": 2,
    "test_adagrad_lr_decay_weight_decay_foreach_xpu": 2,
    "test_adagrad_weight_decay_foreach_xpu": 2,
    "test_adagrad_weight_decay_maximize_foreach_xpu": 2,
    "test_adagrad_tensor_lr_cpu": 6,
    "test_adagrad_tensor_lr_cuda": 6,
    "test_adagrad_tensor_lr_xpu": 6,
    "test_adamax_tensor_lr_weight_decay_capturable_cuda": 6,
    "test_adamax_tensor_lr_weight_decay_capturable_xpu": 6,
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_cuda": 5,
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_xpu": 8,
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_foreach_cuda": 4,
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_foreach_xpu": 4,
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_cuda": 6,
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_xpu": 9,
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_foreach_cuda": 3,
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_foreach_xpu": 3,
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_cuda": 6,
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_xpu": 6,
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_foreach_cuda": 3,
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_foreach_xpu": 3,
    "test_sgd_tensor_lr_cpu": 2,
    "test_sgd_tensor_lr_cuda": 2,
    "test_sgd_tensor_lr_xpu": 2,
    "test_sgd_tensor_lr_foreach_cuda": 2,
    "test_sgd_tensor_lr_foreach_xpu": 2,
}

# also tracks currently supported optimizers
KERNEL_COUNTS = {
    Adam: KernelCounts(multitensor=2, singletensor=8),
    AdamW: KernelCounts(multitensor=2, singletensor=8),
    NAdam: KernelCounts(multitensor=2, singletensor=8),
    Rprop: KernelCounts(multitensor=2, singletensor=8),
    RMSprop: KernelCounts(multitensor=2, singletensor=8),
    Adadelta: KernelCounts(multitensor=2, singletensor=8),
    Adagrad: KernelCounts(multitensor=2, singletensor=8),
    SGD: KernelCounts(multitensor=1, singletensor=8),
    ASGD: KernelCounts(multitensor=2, singletensor=8),
    RAdam: KernelCounts(multitensor=2, singletensor=8),
    Adamax: KernelCounts(multitensor=2, singletensor=8),
}


def build_opt_kwarg_db():
    compiled_opt_db = []
    for optim_info in optim_db:
        if optim_info.optim_cls not in KERNEL_COUNTS:
            continue

        for device in ["cpu", GPU_TYPE]:
            for optim_inputs in _get_optim_inputs_including_global_cliquey_kwargs(
                device, None, optim_info, skip=("differentiable", "fused")
            ):
                kwargs = dict(optim_inputs.kwargs)
                name = f"test_{optim_info.optim_cls.__name__.lower()}"

                has_tensor_lr = False
                for key, val in kwargs.items():
                    if not key == "lr" and (
                        not isinstance(val, bool) or (isinstance(val, bool) and val)
                    ):
                        name += "_" + key

                    if key == "lr" and isinstance(kwargs["lr"], torch.Tensor):
                        has_tensor_lr = True
                        name += "_tensor_lr"

                name += f"_{device}"

                kwargs["device"] = device
                if name in KERNEL_COUNT_OVERRIDES:
                    kwargs["kernel_count"] = KERNEL_COUNT_OVERRIDES[name]
                else:
                    kwargs["kernel_count"] = (
                        KERNEL_COUNTS[optim_info.optim_cls].multitensor
                        if kwargs.get("foreach", False) and device == GPU_TYPE
                        else KERNEL_COUNTS[optim_info.optim_cls].singletensor
                    )

                if kwargs["kernel_count"] is None or kwargs.get("fused", False):
                    continue

                if has_tensor_lr:
                    for scheduler_cls in LR_SCHEDULER_TO_KWARGS.keys():
                        name_w_scheduler = name + f"_{scheduler_cls.__name__.lower()}"
                        compiled_opt_db.append(
                            (
                                optim_info.optim_cls,
                                name_w_scheduler,
                                kwargs,
                                scheduler_cls,
                            )
                        )
                else:
                    compiled_opt_db.append((optim_info.optim_cls, name, kwargs, None))

    return compiled_opt_db


COMPILED_OPT_KWARG_DB = build_opt_kwarg_db()

aten = torch.ops.aten


try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import check_model, check_model_gpu
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


def call_scheduler(scheduler):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(1.0)  # we won't reduce the metric over two iters anyway
    else:
        scheduler.step()


def compile_opt(opt_compiled, closure=None, fullgraph=True):
    # run the patcher so that step has the expected structure
    torch._dynamo.eval_frame.TorchPatcher.patch()

    # unwrap step TWICE to avoid a deliberate graph break due to
    # a limitation of functionalization/no_grad detection
    # see the [Note on graph break] in optimizer.py
    # This ignores the outer _use_grad_if_differentiable wrapper
    # and instead manually disables grad before calling step, which is fine
    # for now as dynamo does not support differentiable optimizers anyway
    step_fn = opt_compiled.step.__wrapped__.__wrapped__

    # This ensures we don't receive spam of warnings from LR Scheduler
    opt_compiled._opt_called = True

    if closure is not None:

        def fn():
            step_fn(opt_compiled, closure)

    else:

        def fn():
            step_fn(opt_compiled)

    return torch.compile(fn, backend="inductor", fullgraph=fullgraph)


def check_optim(
    self,
    optim_cls,
    params_eager,
    params_compiled,
    state_eager,
    state_compiled,
    atol=None,
    rtol=None,
):
    params_eager = list(params_eager)
    params_compiled = list(params_compiled)
    # Note on tolerances:
    # test_correctness_Adadelta_cuda_float32
    # Mismatched elements: 10 / 100 (10.0%)
    # Greatest absolute difference: 4.838220775127411e-05 at index (7, 4) (up to 1e-05 allowed)
    # Greatest relative difference: 0.007270356640219688 at index (7, 2) (up to 1e-05 allowed)
    # This is due to floating point ordering error + usage of sqrt
    rtol = None
    atol = None
    if optim_cls is Adadelta:
        rtol = 5.5e-4
        atol = 5e-5

    self.assertEqual(list(params_eager), list(params_compiled), atol=atol, rtol=rtol)

    for p_eager, p_compiled in zip(params_eager, params_compiled):
        self.assertEqual(
            state_eager[p_eager],
            state_compiled[p_compiled],
            atol=atol,
            rtol=rtol,
        )


def make_test(
    optim_cls,
    closure=None,
    scheduler_cls=None,
    kernel_count=2,
    device="cuda",
    **kwargs,
):
    def test_fn(self):
        stack = ExitStack()
        try:
            # https://github.com/pytorch/pytorch/issues/118715 for capturable Adagrad support
            # https://github.com/pytorch/pytorch/issues/118018 for capturable SGD support
            run_cudagraphs = device == "cuda" and optim_cls not in (Adagrad, SGD)
            if run_cudagraphs:
                stack.enter_context(config.patch({"triton.cudagraphs": True}))

            kwargs_compiled = deepcopy(kwargs)
            if isinstance(kwargs.get("lr", None), torch.Tensor):
                kwargs["lr"] = kwargs["lr"].to(device)
                kwargs_compiled["lr"] = kwargs_compiled["lr"].to(device)

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
            opt_compiled = optim_cls(model_compiled.parameters(), **kwargs_compiled)
            compiled_step = compile_opt(opt_compiled, closure=closure)

            if scheduler_cls:
                scheduler_compiled = create_scheduler(scheduler_cls, opt_compiled)
                scheduler_eager = create_scheduler(scheduler_cls, opt_eager)
                # some schedulers only change after at least an epoch has passed
                scheduler_compiled.last_epoch = 1
                scheduler_eager.last_epoch = 1

            with torch.set_grad_enabled(False):
                for i in range(2):
                    compiled_step()
                    opt_eager.step()
                    if scheduler_cls:
                        call_scheduler(scheduler_eager)
                        call_scheduler(scheduler_compiled)

            check_optim(
                self,
                optim_cls,
                model_eager.parameters(),
                model_compiled.parameters(),
                opt_eager.state,
                opt_compiled.state,
            )

            if run_cudagraphs:
                self.check_cudagraphs_ran()

            if self.check_kernel_count:
                # currently, we compile the step and the rest of the computation
                # separately because the step is a single element tensor
                # hence, the usual kernel count is 2
                self.assertEqual(
                    torch._inductor.metrics.generated_kernel_count, kernel_count
                )
        finally:
            stack.close()

    if device == GPU_TYPE:
        test_fn = requires_gpu(test_fn)

    return test_fn


def make_recompile_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    @requires_gpu
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device=GPU_TYPE)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=GPU_TYPE) for _ in range(2)]
        )
        model(input).sum().backward()

        opt_compiled = optim_cls(model.parameters(), **kwargs)
        compiled_step = compile_opt(opt_compiled)

        # check no recompile here
        with torch.set_grad_enabled(False):
            for _ in range(4):
                compiled_step()

            # perturb state to force recompile
            # Adagrad doesn't reinitialize state on each step
            # SGD has an empty state
            if optim_cls in (Adagrad, SGD):
                opt_compiled.param_groups[0]["lr"] = 0.02
            elif optim_cls is Adam:  # ensure we are guarding on the data_ptr of states
                state_tensor = opt_compiled.state[
                    opt_compiled.param_groups[0]["params"][0]
                ]["exp_avg"]
                opt_compiled.state[opt_compiled.param_groups[0]["params"][0]][
                    "exp_avg"
                ] = torch.zeros_like(state_tensor)
            else:
                opt_compiled.state.clear()

            compiled_step()

        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            # multiply by 2 to account for the recompile
            multiplier = 2

            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count,
                multiplier * kernel_count,
            )

    return test_fn


class CompiledOptimizerParityTests(TestCase):
    @skipCUDAIf(not has_triton(), "torch.compile with cuda requires triton")
    @skipXPUIf(not has_triton(), "torch.compile with xpu requires triton")
    @optims(optim_db, dtypes=[torch.float32])
    @parametrize("use_closure", [True, False])
    def test_correctness(self, device, dtype, optim_info, use_closure):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        if optim_info.step_requires_closure and not use_closure:
            return

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            use_scheduler = isinstance(kwargs.get("lr", None), torch.Tensor)
            scheduler_classes = (
                list(LR_SCHEDULER_TO_KWARGS.keys()) if use_scheduler else [None]
            )

            for scheduler_cls in scheduler_classes:
                torch._dynamo.reset()
                torch._inductor.metrics.reset()
                input = torch.ones([10, 10], device=device)
                model_eager = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10, device=device) for _ in range(2)]
                )
                model_eager(input).sum().backward()
                model_compiled = deepcopy(model_eager)
                model_compiled(input).sum().backward()

                if optim_cls is SparseAdam:
                    for param in model_eager.parameters():
                        param.grad = param.grad.to_sparse()
                    for param in model_compiled.parameters():
                        param.grad = param.grad.to_sparse()

                opt_compiled = optim_cls(
                    model_compiled.parameters(), **deepcopy(kwargs)
                )
                opt_eager = optim_cls(model_eager.parameters(), **deepcopy(kwargs))
                if scheduler_cls:
                    scheduler_compiled = create_scheduler(scheduler_cls, opt_compiled)
                    scheduler_eager = create_scheduler(scheduler_cls, opt_eager)
                    # some schedulers only change after at least an epoch has passed
                    scheduler_compiled.last_epoch = 1
                    scheduler_eager.last_epoch = 1

                num_steps = 2
                if use_closure:

                    @torch.compile()
                    def fn():
                        def closure():
                            loss = model_compiled(input).sum()
                            loss.backward()
                            if optim_info.only_supports_sparse_grads:
                                for param in model_compiled.parameters():
                                    param.grad = param.grad.to_sparse()
                            return loss

                        opt_compiled.step(closure)
                        if scheduler_cls:
                            call_scheduler(scheduler_compiled)

                    def closure_eager():
                        loss = model_eager(input).sum()
                        loss.backward()
                        if optim_info.only_supports_sparse_grads:
                            for param in model_eager.parameters():
                                param.grad = param.grad.to_sparse()

                        return loss

                    for _ in range(num_steps):
                        opt_eager.step(closure_eager)
                        if scheduler_cls:
                            call_scheduler(scheduler_eager)
                else:

                    @torch.compile()
                    def fn():
                        opt_compiled.step()
                        if scheduler_cls:
                            call_scheduler(scheduler_compiled)

                    for _ in range(num_steps):
                        opt_eager.step()
                        if scheduler_cls:
                            call_scheduler(scheduler_eager)

                for _ in range(num_steps):
                    fn()

                check_optim(
                    self,
                    optim_cls,
                    model_eager.parameters(),
                    model_compiled.parameters(),
                    opt_eager.state,
                    opt_compiled.state,
                )


class CompiledOptimizerTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def check_cudagraphs_ran(self):
        # We run the zeroth device currently
        manager = torch._inductor.cudagraph_trees.get_container(0).tree_manager
        self.assertIsNotNone(manager)
        self.assertEqual(manager.new_graph_id().id, 1)

    test_adam_recompile = make_recompile_test(Adam, lr=0.01)
    test_adamw_recompile = make_recompile_test(AdamW, lr=0.01)
    test_adamax_recompile = make_recompile_test(Adamax, lr=0.01)
    test_nadam_recompile = make_recompile_test(NAdam, lr=0.01)
    test_rprop_recompile = make_recompile_test(Rprop, lr=0.01, kernel_count=2)
    test_rmsprop_recompile = make_recompile_test(RMSprop, lr=0.01)
    test_adadelta_recompile = make_recompile_test(Adadelta, lr=0.01)
    test_adagrad_recompile = make_recompile_test(Adagrad, lr=0.01)
    test_asgd_recompile_default = make_recompile_test(ASGD, lr=0.01)
    test_asgd_recompile_single = make_recompile_test(
        ASGD, kernel_count=8, lr=0.01, foreach=False
    )
    test_asgd_recompile_foreach = make_recompile_test(ASGD, lr=0.01, foreach=True)
    test_sgd_recompile_single = make_recompile_test(
        SGD, kernel_count=4, lr=0.01, foreach=False
    )
    test_sgd_recompile_foreach = make_recompile_test(
        SGD, kernel_count=1, lr=0.01, foreach=True
    )

    @requires_gpu
    def test_static_address_finalizer(self):
        import gc

        gc.disable()
        p_ref = None

        def fn():
            nonlocal p_ref
            mod = torch.nn.Linear(10, 10, device=GPU_TYPE, bias=False)
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

    def test_guard_on_none_grads(self):
        def training_loop():
            input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)

            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            )

            params = list(model.parameters())
            optimizer = torch.optim.Adam(params)
            step_list = []

            for i in range(6):
                optimizer.zero_grad()
                # Test that step behaves as expected (a no-op) when grads are set to None
                if i != 3:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                optimizer.step()
                step_list.append(optimizer.state[params[0]]["step"])

            return step_list

        compiled_training_loop = torch._dynamo.optimize("eager")(training_loop)
        actual_steps = compiled_training_loop()
        expected_steps = training_loop()
        self.assertEqual(actual_steps, expected_steps)

    # Basic shampoo test to verify we support compiling the various ops without error
    @requires_gpu
    def test_basic_shampoo(self):
        param_buf = torch.rand((1024, 128))
        param_buf_c = param_buf.clone().detach()

        params_c = [param_buf_c[0:512, :].t(), param_buf_c[512:, :].t()]
        params = [param_buf[0:512, :].t(), param_buf[512:, :].t()]

        for p, p_c in zip(params, params_c):
            p.grad = torch.rand_like(p)
            p_c.grad = p.grad.clone().detach()

        # note this skips the root inverse because this has a lot of internal dependencies
        # we also don't compile it regardless
        @torch.no_grad()
        def shampoo_functional_basic(params):
            step = 1
            weight_decay = 0.1
            grads = [p.grad for p in params]
            beta1 = 0.9
            beta2 = 1.0
            epsilon = 1e-10
            preconditioners = [torch.zeros_like(p) for p in params]
            lr = 0.01

            # pt2 region 1
            # weight decay
            torch._foreach_add_(grads, params, alpha=weight_decay)

            # update preconditioners
            torch._foreach_addcmul_(preconditioners, grads, grads, value=1.0)

            torch._foreach_mul_(grads, beta1)
            torch._foreach_add_(
                grads,
                grads,
                alpha=1 - beta1,
            )
            bias_correction1 = 1.0 - beta1**step
            grad_list = torch._foreach_div(grads, bias_correction1)

            # pt2 region 2
            # precondition (with shampoo branch), with no grafting
            bias_correction2 = 1.0 - beta2**step
            bias_corrected_preconditioner_list = torch._foreach_div(
                preconditioners, bias_correction2
            )
            torch._foreach_sqrt_(bias_corrected_preconditioner_list)
            torch._foreach_add_(bias_corrected_preconditioner_list, epsilon)
            search_directions = torch._foreach_div(
                grad_list, bias_corrected_preconditioner_list
            )

            torch._foreach_add_(
                search_directions,
                params,
                alpha=weight_decay,
            )

            torch._foreach_mul_(search_directions, -lr)
            # pt2 region 3 update params
            torch._foreach_add_(params, search_directions)

            return params, preconditioners, grads

        compiled_fn = torch.compile(shampoo_functional_basic)

        self.assertEqual(compiled_fn(params_c), shampoo_functional_basic(params))

    @requires_gpu
    def test_closure_graph_break(self):
        param = torch.rand(
            2, 3, dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        param_c = param.clone().detach().requires_grad_(True)

        def closure():
            param.grad = torch.ones_like(param) * 2
            return param.grad

        def closure_c():
            param_c.grad = torch.ones_like(param_c) * 2
            return param_c.grad

        optimizer = torch.optim.AdamW([param])
        optimizer_c = torch.optim.AdamW([param_c])

        def loop(opt, c):
            opt.step(c)

        compiled_loop = torch._dynamo.optimize("eager")(loop)

        compiled_loop(optimizer, closure)
        loop(optimizer_c, closure_c)

        self.assertEqual(param, param_c)

    def test_get_value_on_static_address(self):
        from torch._dynamo.decorators import mark_static_address
        from torch.optim.optimizer import _get_value

        compiled = torch.compile(_get_value)

        x = torch.ones(2, 2)
        mark_static_address(x)

        ret_val = compiled(x)

        self.assertEqual(ret_val, x)

    # compile a large foreach op and verify
    # that the time taken is within an expected range
    @requires_gpu
    def test_compile_time_smoketest(self):
        import time

        xs = [torch.ones(2, 2, device=GPU_TYPE) for _ in range(100)]
        ys = [torch.ones(2, 2, device=GPU_TYPE) for _ in range(100)]

        @torch.compile
        def fn(xs, ys):
            return torch._foreach_add(xs, ys)

        start = time.perf_counter()
        fn(xs, ys)
        end = time.perf_counter()

        self.assertLess(end - start, 90)

    @requires_cuda
    def test_S429861(self):
        # Just verify we can compile this function without error
        try:
            from . import s429861_repro
        except ImportError:
            import s429861_repro

        forward = s429861_repro.forward

        import torch._dynamo
        import torch._inductor
        from torch._dynamo.debug_utils import aot_graph_input_parser
        from torch._inductor.utils import fresh_inductor_cache

        with fresh_inductor_cache():
            kwargs = aot_graph_input_parser(forward)
            torch.compile(forward)(**kwargs)


for optim_cls, name, kwargs, scheduler_cls in COMPILED_OPT_KWARG_DB:
    setattr(
        CompiledOptimizerTests,
        name,
        make_test(optim_cls, scheduler_cls=scheduler_cls, **kwargs),
    )

instantiate_device_type_tests(
    CompiledOptimizerParityTests, globals(), allow_xpu=True, except_for="cpu"
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
