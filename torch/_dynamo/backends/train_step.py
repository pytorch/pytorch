from typing import List

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch._dynamo import register_backend
from torch._dynamo.backends.registry import lookup_backend
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.interpreter import Interpreter
from torch.nn.utils import stateless


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
        print(mod.graph)
        torch.set_grad_enabled(True)
        torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)
        assert len(fake_inputs) > 0, "Expected at least one input"
        fake_mode = fake_inputs[0].fake_mode
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "Expected a valid FakeTensorMode on dynamo inputs"

        def fakeify_inputs(flat_args):
            def convert(idx, x):
                # todo: do we expect symint inputs?
                assert isinstance(x, torch.Tensor)
                return fake_mode.from_tensor(x, static_shapes=False)

            return [convert(idx, x) for idx, x in enumerate(flat_args)]

        params = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params_len = len(params_flat)
        fake_params_flat = fakeify_inputs(params_flat)
        full_fake_args = fake_params_flat + fake_inputs

        """
        Step 2: Create a new graphmodule that accepts parameters and user-inputs
                as inputs to forward(), instead of accessing parameters as attrs.
        """

        def functional_call(*args, **kwargs):
            with stateless._reparametrize_module(
                mod, pytree.tree_unflatten(args[:params_len], params_spec)
            ):
                out = mod(*args[params_len:], **kwargs)

            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "Graph output must be a tuple() to avoid pytree processing of the ouputs."
                )
            return out

        # This fx_g contains expanded backward ops, but also accepts flat params as inputs to forward, so we can't
        # directly return it
        assert torch.is_grad_enabled(), "grad isn't enabled before calling make_fx"
        fx_g = make_fx(functional_call)(*full_fake_args)
        torch.set_grad_enabled(False)
        print("fx_g")
        print(fx_g)
        """
        Step 3: Functionalize the resulting flattend graph, producing code with copy_ ops
                as an epilogue for any inplace/mutating ops such as optimizer update.
        """

        def retraced_f(*args):
            return Interpreter(fx_g).run(*args)

        # not really sure why we need inference mode here
        with torch.inference_mode():
            functional_fx_g = make_fx(functionalize(retraced_f))(*full_fake_args)

        """
        Step 4: Reverse the calling-convention change we made above with _reparametrize_module,
                and return a function that accepts the arguments as originally provided by dynamo
        """
        print("functional_fx_g.graph")
        print(functional_fx_g.graph)

        def call_without_params(*runtime_args):
            with torch.no_grad():
                return functional_fx_g(*params_flat + list(runtime_args))

        return call_without_params

    return _compile_fn


train_step_eager = train_step_compiler(lookup_backend("eager"))
register_backend(name="train_step_eager", compiler_fn=train_step_eager)
