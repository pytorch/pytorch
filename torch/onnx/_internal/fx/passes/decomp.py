# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib

from typing import Callable, Mapping, Optional

import torch
import torch._ops
import torch.fx
from torch._dispatch import python as python_dispatch
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics
from torch.onnx._internal.fx.passes import _utils


class Decompose(_pass.Transform):
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        decomposition_table: Mapping[torch._ops.OpOverload, Callable],
        enable_dynamic_axes: bool,
        allow_fake_constant: Optional[bool] = False,
    ):
        super().__init__(diagnostic_context, module)
        self.decomposition_table = decomposition_table
        self.enable_dynamic_axes = enable_dynamic_axes
        self.allow_fake_constant = allow_fake_constant

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert not kwargs, "kwargs is not supported in Decompose."

        # To preserve stack trace info after `make_fx`.
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)

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

        # Mimic `torch._dynamo.export(aten_graph=True)` behavior in invoking `make_fx`.
        # TODO: May need revisit for user fake mode export + dynamic shape scenario.
        fake_mode: Optional[fake_tensor.FakeTensorMode] = self.fake_mode
        maybe_fake_args = self._maybe_fakefy_args(fake_mode, *args)
        if fake_mode is not None:
            # Using existing fake mode as context, signal `make_fx` that it does not need
            # to create a new fake mode by passing tracing_mode as "real".
            tracing_mode = "real"
        else:
            # Existing fake mode not found, signal `make_fx` to create one.
            fake_mode = contextlib.nullcontext()  # type: ignore[assignment]
            tracing_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        # Apply decomposition table to the input graph.
        assert fake_mode is not None  # for mypy
        with proxy_tensor.maybe_disable_fake_tensor_mode(), python_dispatch.enable_python_dispatcher(), (
            fake_mode
        ):
            decomposed_module = proxy_tensor.make_fx(
                module,
                decomposition_table=self.decomposition_table,
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=True,
                _allow_fake_constant=self.allow_fake_constant,
            )(*maybe_fake_args)

        # Rename placeholder targets to match the original module's signature since
        # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
        _utils.replace_placeholder_name_and_target(decomposed_module, self.module)

        return decomposed_module
