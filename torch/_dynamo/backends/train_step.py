from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Dict, List
from unittest import mock

import torch
from torch._dynamo.utils import assert_no_fake_params_or_buffers, check_all_fake
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
import torch.utils._pytree as pytree
from torch import fx
from torch._dynamo import register_backend
from torch._dynamo.backends.registry import lookup_backend
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, PythonKeyTracer, make_fx
from torch.fx.interpreter import Interpreter
from torch._guards import detect_fake_mode
from torch.nn.utils import stateless

import torch.utils._pytree as pytree

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

def get_deferred_modes():
    fx_tracer = PythonKeyTracer()
    fx_tracer.graph = Graph(fx_tracer)
    fx_tracer.root = torch.nn.Module()
    fx_tracer.tensor_attrs = {}
    proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode="real")
    return proxy_mode, fx_tracer


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

    def _compile_fn(mod: fx.GraphModule, real_inputs: List[torch.Tensor]):
        """
        Step 1: Assert inputs (from user) are already Fake, and user their FakeTensorMode
                (created by dynamo) to fakeify the module's parameters
        """
        assert (
            torch.is_grad_enabled()
        ), "Expected grad enabled when calling train_step_compile"
        if check_all_fake(mod):
            deferred_init = True
        else:
            deferred_init = False
            assert_no_fake_params_or_buffers(mod)
        assert len(real_inputs) > 0, "Expected at least one input"
        fake_mode = detect_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "Expected a valid FakeTensorMode"

        def fakeify_tensors(flat_args):
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
        fake_params_flat = params_flat if deferred_init else fakeify_tensors(params_flat)
        fake_inputs = fakeify_tensors(real_inputs)
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

            # In order to trace optimizer _init_group, param grads must exist.
            # But if I run the train_step graph to make param grads exist, it also initializes the optimizer
            # which prevents re-initializing the optimizer during optimizer tracing.
            with fake_mode:
                for param in fake_params_flat:
                    param.grad = torch.empty_like(param)

            optimizer_proxy_mode, optimizer_fx_tracer = get_deferred_modes()
            with fake_mode, optimizer_proxy_mode:
                breakpoint()
                for group in opt.param_groups:
                    params_with_grad = []
                    grads = []
                    exp_avgs = []
                    exp_avg_sqs = []
                    max_exp_avg_sqs = []
                    state_steps = []

                    opt._init_group(
                        group,
                        params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps)
                # Convert the fake optimizer states to real
                outputs = []
                for param in fake_params_flat:
                    assert param in opt.state, "all params expected handled by one optimizer, multi-opt NYI"
                    for name, state in opt.state[param].items():
                        if hasattr(state, "proxy"):
                            print(f"Has proxy: {name} {state}")
                            outputs.append(state.proxy.node) 
                        else:
                            print(f"Missing proxy: {name} {state}")

                optimizer_fx_tracer.graph.output(outputs)
                optimizer_fx_tracer.graph.eliminate_dead_code()  # hmmm
                opt_deferred_init = GraphModule(optimizer_fx_tracer.root, optimizer_fx_tracer.graph)
                results = opt_deferred_init()
                breakpoint()
                print()


            # Build a mapping to use for reparametrizing the optimizer during tracing
            named_states = {}
            for n, p in pytree.tree_unflatten(fake_params_flat, params_spec).items():
                if p in opt.state:
                    named_states[n] = opt.state[p]  # type: ignore[index]

        named_states_flat, named_states_spec = pytree.tree_flatten(named_states)
        fake_named_states_flat = fakeify_tensors(named_states_flat)
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
        Step 4: Call the backend compiler
        """


        """
        Step 5: Make the model 'real' if it was deferred
        """
        results_iter = iter(mod.model._deferred_init())

        for i, t in enumerate(params_flat):
            if t is None:
                continue
            params_flat[i] = next(results_iter)

        
        """
        Step 6: Reverse the calling-convention change we made above with _reparametrize_module,
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
