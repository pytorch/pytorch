import builtins
import logging
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch import _C, fx
from torch._dynamo.backends.registry import lookup_backend
from torch._guards import detect_fake_mode, TracingContext
from torch._inductor.compile_fx import compile_fx_inner

from torch._inductor.decomposition import select_decomp_table
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from .is_train_step import TrainStepCompiler

log = logging.getLogger(__name__)


def _compile_train_step(
    train_step_fn: Callable,
    *,
    dynamic: builtins.bool = False,
    backend: Union[str, Callable] = "inductor",
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> Callable:
    """
    Compiles a whole train step function, without graph-breaking on .backward().

    EXPERIMENTAL: both how the API is constructed and how it behaves are experimental and subject to change.

    Limitations:
    - No optimizer support yet
    - All inputs to the train_step fn (whether args/kwargs or globals) must not have .grad_fn set, meaning
      you may not use these tensors in gradient-requiring operations outside of the compiled region

    _compile_train_step args are copied from `torch.compile` so see those docs for more info.
    - note: fullgraph=True is implied

    Example:

        from torch._dynamo.backends._train_step import _compile_train_step

        def train_step(model, inputs):
            ...

        opt_train_step = _compile_train_step(train_step, ...)
    """
    _C._log_api_usage_once("torch._dynamo.backends.train_step._compile_train_step")

    import torch._dynamo

    if not _is_train_step_compiler(backend):
        raise RuntimeError(
            f"_compile_train_step does not support {backend}, which is not wrapped in TrainStepCompiler."
            " If you want to make a new TrainStepCompiler backend, ensure your backend does not call aot_autograd,"
            " and wrap it in TrainStepCompiler."
        )

    return torch._dynamo.optimize(
        backend=backend, nopython=True, dynamic=dynamic, disable=disable
    )(train_step_fn)


def _train_step_compiler(backend_compile_fn):
    """Note [Train Step Compile]

    Usually, torch.compile() allows graph-breaks and compiles pairs of forward (+backward) by
    extracting sections of forward from python programs and using AotAutograd to produce corresponding
    chunks of backwards, tying it back together with an AotFunction.

    Instead, TrainStepCompiler assumes the user compiles a full train_step function complete with calls to
    .backward(), and zero_grad().  It additionally requires no graph-breaks.

    Args:
        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.
    """
    is_inductor = backend_compile_fn is compile_fx_inner
    backend_decomps = select_decomp_table() if is_inductor else None

    def _compile_fn(mod: fx.GraphModule, real_inputs: List[torch.Tensor]):
        """
        Step 1: Assert inputs (from user) are already Fake, and user their FakeTensorMode
                (created by dynamo) to fakeify the module's parameters
        """
        assert (
            torch.is_grad_enabled()
        ), "Expected grad enabled when calling train_step_compile"
        torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)
        assert len(real_inputs) > 0, "Expected at least one input"
        fake_mode = detect_fake_mode()

        tc = TracingContext.train_step_context(assert_if_missing=True)
        assert isinstance(fake_mode, FakeTensorMode), "Expected a valid FakeTensorMode"

        def fakeify_inputs(flat_args):
            already_fake = {}

            def convert(idx, x):
                # todo: do we expect symint inputs?
                assert isinstance(x, torch.Tensor)
                if x not in already_fake:
                    # Since we do have duplicate names from dynamo refering to the same tensor,
                    # ensure that we never make more than one faketensor for a given real tensor!
                    already_fake[x] = fake_mode.from_tensor(x, static_shapes=False)
                return already_fake[x]

            return [convert(idx, x) for idx, x in enumerate(flat_args)]

        params = {
            **dict(mod.named_parameters()),
            **dict(mod.named_buffers()),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params_len = len(params_flat)
        fake_params_flat = fakeify_inputs(params_flat)
        fake_inputs = fakeify_inputs(real_inputs)

        log.debug("\n---original graph---\n%s\n\n", mod.graph)

        def functional_call(*lifted_args, **kwargs):
            """Call the dynamo graphmodule in a functional way safe for tracing
            (lifts module parameters and optimizer states as inputs)
            """
            _params = lifted_args[:params_len]
            _params_dict = pytree.tree_unflatten(_params, params_spec)
            _user_args = lifted_args[params_len:]
            with stateless._reparametrize_module(mod, _params_dict):
                out = mod(*_user_args, **kwargs)

            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "Graph output must be a tuple() to avoid pytree processing of the ouputs."
                )
            return out

        """
        Step 1: Warm up the optimizer(s) (if present).
        """
        # TODO support optimizers
        full_fake_args = fake_params_flat + fake_inputs

        """
        Step 2: Trace the full graph, invoking backend-specific decomps.
                Expand the .backward() and .step/.zero_grad calls into aten ops.
        """
        with fake_mode:
            fx_g = make_fx(functional_call, decomposition_table=backend_decomps)(
                *full_fake_args
            )
        log.debug("\n---functional_call graph---\n%s\n\n", fx_g.graph)

        """
        Step 3: Functionalize the resulting flattend graph, producing code with copy_ ops
                as an epilogue for any inplace/mutating ops such as optimizer update.
        """
        with fake_mode, torch.inference_mode():
            # We need to disable grad, since we will be inplace-updating leaf nodes (optimizer acting on params)
            functional_fx_g = make_fx(functionalize(fx_g))(*full_fake_args)
            log.debug("\n---functionalized graph---\n%s\n\n", functional_fx_g.graph)

        """
        Step 4: Call the user compiler. This user compiler must be aware/capable of supporting a full train graph,
                and should not attempt to call AOTAutograd internally.
        """
        with torch.inference_mode():
            backend_fx_g = backend_compile_fn(functional_fx_g, full_fake_args)

        """
        Step 5: Reverse the calling-convention change we made above with _reparametrize_module,
                and return a function that accepts the arguments as originally provided by dynamo.
        """

        def call_without_params(*runtime_args):
            with torch.inference_mode():
                # See note above about disabling grad
                # TODO can this divergence be unified?
                _args = params_flat + list(runtime_args)
                if is_inductor:
                    return backend_fx_g(_args)
                else:
                    return backend_fx_g(*_args)

        return call_without_params

    return TrainStepCompiler(_compile_fn)


_train_step_inductor = _train_step_compiler(compile_fx_inner)
_train_step_eager = _train_step_compiler(lookup_backend("eager"))


def _is_train_step_compiler(compiler_fn: Callable):
    return isinstance(compiler_fn, TrainStepCompiler)
