from contextlib import contextmanager
from copy import copy
from typing import Any, Dict, List

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
    assert opt is not None

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
    print(f"setting optimizer to use parm0 {params}")
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
        # at input dynamo puts two copies e.g.
        #   L__model___layers_0.weight
        #   model.layers.0.weight
        # both as realtensor on passed in graphmodule.  `model` comes from having model as input to train_step.

        print(mod.graph)
        torch.set_grad_enabled(True)
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

        # TODO fix dynamo's "list of optimizers" on the module, and iterate them instead of hardcoding here
        opt = mod.__optimizer_0


        def functional_call(*lifted_args, **kwargs):
            """Call the dynamo graphmodule in a functional way safe for tracing
            (lifts module parameters and optimizer states as inputs)
            """

            _params = lifted_args[:params_len]
            _params_dict = pytree.tree_unflatten(_params, params_spec)
            _user_args = lifted_args[params_len + named_states_len :]
            with stateless._reparametrize_module(
                mod, _params_dict
            ), _rematerialize_optimizer(opt, None, _params_dict):
                out = mod(*_user_args, **kwargs)

            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "Graph output must be a tuple() to avoid pytree processing of the ouputs."
                )
            return out

        """
        Step 1: Warm up the optimizer
        - this adds state tensors to the previously empty optimizer state dict

        """
        named_states_len = 0
        # _ = functional_call(*fake_params_flat + fake_inputs)
        dev = params_flat[0].device
        # so we don't mutate the real params when running the warmup...
        # copied_params = [p.clone().detach() for p in params_flat]
        # running with fake inputs and fixing-up the opt states is hard, since the opt-states
        # get keyed off _mutated_ faketensor module params, which have diff ids than orig fake module params
        # real_inputs = [
        #     torch.randn(i.shape, dtype=i.dtype, device=dev) for i in fake_inputs
        # ]

        fake_mode.allow_non_fake_inputs = True
        first_loss = functional_call(*fake_params_flat + fake_inputs)
        fake_mode.allow_non_fake_inputs = False

        # Convert the fake optimizer states to real
        for fake_param, state_dict in opt.state.items():
            for name, state in state_dict.items():
                # some of the states are singleton cpu tensors...
                if isinstance(state, FakeTensor):
                    # can we assume always init with zeros?
                    state_dict[name] = torch.zeros(state.shape, dtype=state.dtype, device=dev)
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
        Step 2: Trace the full graph
        """

        def functional_call_2(*lifted_args, **kwargs):
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

        fx_g = make_fx(functional_call_2)(*full_fake_args)
        torch.set_grad_enabled(False)
        # print(f"fx_g: {fx_g}")

        """
        Step 3: Functionalize the resulting flattend graph, producing code with copy_ ops
                as an epilogue for any inplace/mutating ops such as optimizer update.
        """

        def retraced_f(*args):
            return Interpreter(fx_g).run(*args)

        with torch.inference_mode():
            functional_fx_g = make_fx(functionalize(retraced_f))(*full_fake_args)
            # print(f"functional_fx_g.graph {functional_fx_g.graph}")

        """
        Step 4: Reverse the calling-convention change we made above with _reparametrize_module,
                and return a function that accepts the arguments as originally provided by dynamo
        """

        def call_without_params(*runtime_args):
            with torch.no_grad():
                return functional_fx_g(
                    *params_flat + named_states_flat + list(runtime_args)
                )

        return call_without_params

    return _compile_fn


train_step_eager = train_step_compiler(lookup_backend("eager"))
register_backend(name="train_step_eager", compiler_fn=train_step_eager)
