from __future__ import annotations

import contextlib

from typing import Callable, Dict, List, Any

import torch
import torch._ops
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental import proxy_tensor
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass


@_beartype.beartype
def _rename_placeholder_targets(
    module: "torch.fx.GraphModule", reference_module: "torch.fx.GraphModule"
):
    """Align the argument names in module with those in reference_module.
    After calling this function, the two forward(...) in module and reference_module should have
    the same signature.
    """
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    reference_placeholders = [
        node for node in reference_module.graph.nodes if node.op == "placeholder"
    ]

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        placeholder.name = reference_placeholder.name

    module.recompile()


def wrapped_fn(fn: Callable, mode: str, allow_non_fake_inputs: bool, args: List[Any]):
    context = enable_python_dispatcher if mode == "symbolic" else contextlib.nullcontext
    shape_env = ShapeEnv() if mode == "symbolic" else None
    fake_mode = torch._dynamo.utils.fake_mode_from_tensors(args)
    if mode is None:
        fake_mode = FakeTensorMode(
            shape_env=shape_env, allow_non_fake_inputs=allow_non_fake_inputs
        )

    args = [
        (fake_mode.from_tensor(t)
        if isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor)
        else t)
        for t in args
    ]

    def fn_wrapped():
        with context():
            with fake_mode:
                return fn(*args)

    return fn_wrapped


class Decompose(_pass.Transform):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        decomposition_table: Dict[torch._ops.OpOverload, Callable],
        enable_dynamic_axes: bool,
    ):
        super().__init__(module)
        self.decomposition_table = decomposition_table
        self.enable_dynamic_axes = enable_dynamic_axes

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert not kwargs, "kwargs is not supported in Decompose."
        # A trick adopted from `dynamo.export` in `eval_frame.py`.
        # Running graph with interpreter is needed for propagating the stack_trace.

        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(self.module).run(*args)

        # fake mode use static size to trace the size of tensors. while symbolic
        # mode generates aten::sym_size to dynamically trace the size of tensors.

        # e.g. fake mode:
        #  view: f32[3, 5, 20] = torch.ops.aten.view.default(x, [3, 5, 20])

        # e.g. symbolic mode:
        #  sym_size = torch.ops.aten.sym_size(x, 0)
        #  sym_size_1 = torch.ops.aten.sym_size(x, 1)
        #  sym_size_2 = torch.ops.aten.sym_size(x, 2)
        #  sym_size_3 = torch.ops.aten.sym_size(x, 3)
        #  mul = sym_size_2 * sym_size_3;  sym_size_2 = sym_size_3 = None
        #  view: f32[3, 5, 20] = torch.ops.aten.view.default(x, [sym_size, sym_size_1, mul])

        fx_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        fn = wrapped_fn(
            graph_with_interpreter, fx_mode, allow_non_fake_inputs=True, args=args
        )

        # Apply decomposition table to the input graph.
        # Make sure the feed-in "module" is stateless.
        decomposed_module = proxy_tensor.make_fx(
            fn,
            decomposition_table=self.decomposition_table,
            tracing_mode="real",
            _allow_non_fake_inputs=True,
        )()
        # Rename placeholder targets to match the original module's signature since
        # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
        _rename_placeholder_targets(decomposed_module, self.module)

        return decomposed_module
