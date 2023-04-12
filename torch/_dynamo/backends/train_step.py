from contextlib import contextmanager
from copy import copy
from typing import Any, Dict, List
from unittest import mock

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch._dynamo import register_backend
from torch._dynamo.backends.registry import lookup_backend
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.interpreter import Interpreter
from torch.nn.utils import stateless


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, torch.nn.Parameter],
):
    if opt is None:
        try:
            yield
        finally:
            pass
        return

    # update opt.state with proxy tensors
    orig_states: Dict[str, Any] = copy(opt.state)
    if named_states:
        for n in named_states:
            # opt.state's key type is string, but optimizer uses Parameter as keys
            opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    # FIXME(@mrshenli): exclude buffers
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state.update(orig_states)


def train_step_compiler(backend_compile_fn):
    """Note [Train Step Compile]

    Usually, torch.compile() allows graph-breaks and compiles pairs of forward (+backward) by
    extracting sections of forward from python programs and using AotAutograd to produce corresponding
    chunks of backwards, tying it back together with an AotFunction.

    Instead, TrainStepCompiler assumes the user compiles a full train_step function complete with calls to
    .backward(), optimizer step(), and zero_grad().  It additionally requires no graph-breaks.

    Args:
        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.
    """

    def _compile_fn(mod: fx.GraphModule, fake_inputs: List[torch.Tensor]):
        """
        Step 1: Assert inputs (from user) are already Fake, and user their FakeTensorMode
                (created by dynamo) to fakeify the module's parameters
        """
        assert (
            torch.is_grad_enabled()
        ), "Expected grad enabled when calling train_step_compile"
        torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)
        assert len(fake_inputs) > 0, "Expected at least one input"
        fake_mode = fake_inputs[0].fake_mode
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "Expected a valid FakeTensorMode on dynamo inputs"

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

        # problem: if i don't remove duplicates here, stateless.Parametrize gets confused by presence of duplicates
        # duplicates come from dynamo.  it adds `model` key which contains one set of references to all params/buffers,
        # and then it also adds individual keys for names actually traced by dynamo.
        params = {
            **dict(mod.named_parameters(remove_duplicate=True)),
            **dict(mod.named_buffers(remove_duplicate=True)),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params_len = len(params_flat)
        fake_params_flat = fakeify_inputs(params_flat)

        assert (
            "optimizers" in mod.meta
        ), "Dynamo should populate GraphModule meta with optimizers dict"
        optimizers = mod.meta["optimizers"]
        assert len(optimizers) <= 1, "Multiple optimizers NYI"

        def functional_call(*lifted_args, **kwargs):
            """Call the dynamo graphmodule in a functional way safe for tracing
            (lifts module parameters and optimizer states as inputs)
            """
            _params = lifted_args[:params_len]
            _params_dict = pytree.tree_unflatten(_params, params_spec)
            _named_states = lifted_args[params_len : params_len + named_states_len]
            _named_states_dict = pytree.tree_unflatten(_named_states, named_states_spec)
            _user_args = lifted_args[params_len + named_states_len :]
            with stateless._reparametrize_module(
                mod, _params_dict
            ), _rematerialize_optimizer(opt, _named_states_dict, _params_dict):
                out = mod(*_user_args, **kwargs)

            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "Graph output must be a tuple() to avoid pytree processing of the ouputs."
                )
            return out

        opt = None
        # for the optimizer warmup, we need empty named_states for reparametrize_optimizer,
        # but we want to reuse the same 'functional_call' which looks for this
        named_states = {}
        named_states_flat, named_states_spec = pytree.tree_flatten(named_states)
        named_states_len = len(named_states_flat)

        """
        Step 1: Warm up the optimizer(s) (if present).
        """
        if len(optimizers):
            # TODO iterate properly
            opt = optimizers["__optimizer_0"]
            dev = params_flat[0].device

            # allow_non_fake_inputs
            #
            # the reason for allow_non_fake_inputs is that we are initializing optimizer states implicitly
            # on the first invocation, and the _init_group function creates size-0 tensors with real values
            # which upset FakeTensor otherwise.

            # In practice, the adam optimizer sets its state_dict["step"] values to real tensors
            # which i'm afraid means we aren't quite tracing the program correctly unless we can
            # restore it, which we attempt below with .zero_()

            # Question: can we enforce that 'capturable' is true for the param_groups? this codepath
            # looks like it would avoid this problem entirely but I'm not sure how to set it.
            with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
                # This adds fake state tensors to the previously empty optimizer state dicts.
                _ = functional_call(*fake_params_flat + fake_inputs)

            # Convert the fake optimizer states to real
            for fake_param, state_dict in opt.state.items():
                for name, state in state_dict.items():
                    if isinstance(state, FakeTensor):
                        # we assume always init with zeros, which is lame: can we trace init separately?
                        state_dict[name] = torch.zeros(
                            state.shape, dtype=state.dtype, device=dev
                        )
                    else:
                        # some of the states are singleton cpu tensors, e.g. 'step'...
                        state_dict[name].zero_()

            # Build a mapping to use for reparametrizing the optimizer during tracing
            named_states = {}
            for n, p in pytree.tree_unflatten(fake_params_flat, params_spec).items():
                if p in opt.state:
                    named_states[n] = opt.state[p]  # type: ignore[index]

        named_states_flat, named_states_spec = pytree.tree_flatten(named_states)
        fake_named_states_flat = fakeify_inputs(named_states_flat)
        named_states_len = len(named_states_flat)
        full_fake_args = fake_params_flat + fake_named_states_flat + fake_inputs

        """
        Step 2: Trace the full graph.
        """
        fx_g = make_fx(functional_call)(*full_fake_args)

        """
        Step 3: Functionalize the resulting flattend graph, producing code with copy_ ops
                as an epilogue for any inplace/mutating ops such as optimizer update.
        """
        with torch.inference_mode():
            # We need to disable grad, since we will be inplace-updating leaf nodes (optimizer acting on params)
            functional_fx_g = make_fx(functionalize(fx_g))(*full_fake_args)

        """
        Step 4: Reverse the calling-convention change we made above with _reparametrize_module,
                and return a function that accepts the arguments as originally provided by dynamo.
        """

        def call_without_params(*runtime_args):
            with torch.inference_mode():
                # See note above about disabling grad
                return functional_fx_g(
                    *params_flat + named_states_flat + list(runtime_args)
                )

        return call_without_params

    return _compile_fn


train_step_eager = train_step_compiler(lookup_backend("eager"))
register_backend(name="train_step_eager", compiler_fn=train_step_eager)
