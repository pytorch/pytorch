# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch._ops
import torch.func
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal.fx import _pass, diagnostics
from torch.onnx._internal.fx.passes import _utils
from torch.utils import _pytree as pytree


class Functionalize(_pass.Transform):
    """Functionalize a GraphModule.

    This pass utilizes ``functionalization`` utility of ``torch._functorch`` to convert
    a GraphModule into a functional form. The two main functionalities are (copied from
    its documentations):

    * ``functionalization`` removes (intermediate) mutations and aliasing from a
    function, while preserving the function's semantics.

    * ``functionalization`` also removes mutations (and views) that were performed
    on function inputs. However to preserve semantics, functionalize will "fix up" the
    mutations after the transform has finished running, by detecting if any tensor inputs
    "should have" been mutated, and copying the new data back to the inputs if necessary.
    For example, consider::

        def fn(a, b):
            a.add_(b)
            return a

      For a call like `fn(x, y)`, the variable `x` outside is also mutated. Hence just
      functionalizing is not enough for preserving the original semantics. A "special"
      input mutation step needs to be inserted at the end.::

        # After functionalization, without input mutation "fix up".
        # This is not semantically the same. The variable outside the function call that
        # was passed in as `a` is not mutated.
        def fn(a, b):
            new_a = a + b
            return new_a

        # Functionalization with input mutation "fix up" that preserves semantics.
        def fn(a, b):
            new_a = a + b

            # Copying the new data back to the inputs
            a.copy_(new_a)

            return new_a

    For ONNX inference, it is recommended to run ``RemoveInputMutation`` after this pass.
    ``RemoveInputMutation`` removes the "fix up" nodes that were added by ``Functionalize``,
    which are not needed for ONNX inference.
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        enable_dynamic_axes: bool,
        allow_fake_constant: bool | None = False,
    ):
        super().__init__(diagnostic_context, module)
        self.enable_dynamic_axes = enable_dynamic_axes
        self.allow_fake_constant = allow_fake_constant

    def _functionalize(self, function: Callable) -> Callable:
        # Working around a dispatcher issue with `torch.func.functionalize` when used
        # together with `make_fx`.
        # Ref: https://github.com/pytorch/pytorch/issues/99774#issuecomment-1527949391
        def wrapped(*inputs):
            inputs_functional = pytree.tree_map_only(
                torch.Tensor, torch._to_functional_tensor, inputs
            )
            torch._enable_functionalization(reapply_views=True)
            try:
                out = function(*inputs_functional)
            finally:
                torch._disable_functionalization()

            flat_inputs_functional = pytree.tree_leaves(inputs_functional)
            for input_functional in flat_inputs_functional:
                if isinstance(input_functional, torch.Tensor):
                    torch._sync(input_functional)
            pytree.tree_map(torch._sync, out)
            out_unwrapped = pytree.tree_map(torch._from_functional_tensor, out)
            return out_unwrapped

        return wrapped

    def _run(self, *args) -> torch.fx.GraphModule:
        # To preserve stack trace info after `make_fx`.
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)

        functionalized_callable = self._functionalize(module)

        # Mimic `torch._dynamo.export(aten_graph=True)` behavior in invoking `make_fx`.
        # TODO: May need revisit for user fake mode export + dynamic shape scenario.
        fake_mode: fake_tensor.FakeTensorMode | None = self.fake_mode
        maybe_fake_args = self._maybe_fakefy_args(fake_mode, *args)
        if fake_mode is not None:
            # Using existing fake mode as context, signal `make_fx` that it does not need
            # to create a new fake mode by passing tracing_mode as "real".
            tracing_mode = "real"
        else:
            # Existing fake mode not found, signal `make_fx` to create one.
            fake_mode = contextlib.nullcontext()  # type: ignore[assignment]
            tracing_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        assert fake_mode is not None  # for mypy
        with fake_tensor.unset_fake_temporarily(), fake_mode:
            graph_module = proxy_tensor.make_fx(
                functionalized_callable,
                decomposition_table={},
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=True,
                _allow_fake_constant=bool(self.allow_fake_constant),
            )(*maybe_fake_args)

        # Rename placeholder targets to match the original module's signature since
        # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
        _utils.replace_placeholder_name_and_target(graph_module, self.module)

        return graph_module


class RemoveInputMutation(_pass.Transform):
    """Remove `aten.copy_.default` nodes that mutate module inputs.

    This pass is recommended to be used after ``Functionalization`` pass.
    ``Functionalization`` pass adds `aten.copy_.default` nodes to the graph
    when it detects mutations to inputs. These nodes are not needed for ONNX export
    for inference. They could be useful for training.
    """

    def _run(self, *args) -> torch.fx.GraphModule:
        for node in reversed(self.module.graph.nodes):
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.copy_.default
                and len(node.users) == 0
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].op == "placeholder"
            ):
                self.module.graph.erase_node(node)
        return self.module
