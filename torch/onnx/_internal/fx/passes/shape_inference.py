from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx._compatibility import compatibility
from torch.fx.experimental import proxy_tensor
from torch.fx.node import map_aggregate
from torch.nn.utils import stateless

from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass


class ShapeInferenceWithFakeTensor(_pass.Transform):
    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert not kwargs, "`kwargs` is not supported."
        module = self.module
        # NOTE(titaiwang): Usually fx graph should have all the node meta value we need,
        # so we don't have to run FakeTensorProp to fill in node meta values. However, this
        # can be used to validate op-level debugging when we only have symbolic shapes in
        # graph

        # Use this FakeTensorMode to
        # 1. convert nn.Parameter's in nn.Module to FakeTensor
        # 2. run FakeTensorProp
        # If (1) and (2) are done with difference FakeTensorMode's, undefined behavior may
        # happen.
        fake_tensor_mode = fake_tensor.FakeTensorMode()

        def to_fake_tensor(x):
            if isinstance(x, torch.Tensor) and not isinstance(
                x, fake_tensor.FakeTensor
            ):
                return fake_tensor_mode.from_tensor(x)
            return x

        # "args" are FakeTensor in FakeTensorProp so the parameters and buffers
        # in model must be converted to FakeTensor as well.
        fake_parameters_and_buffers = {
            k: to_fake_tensor(v)
            for k, v in itertools.chain(
                module.named_parameters(), module.named_buffers()
            )
        }
        # Shape inference via FakeTensorProp
        with stateless._reparametrize_module(module, fake_parameters_and_buffers):
            # Assign output types and shapes to each node without meta values.
            ValidationFakeProp(module, fake_tensor_mode).propagate(*args)
        return module


@compatibility(is_backward_compatible=False)
class ValidationFakeProp(torch.fx.Interpreter):
    """
    This is heavily referenced from torch.fx.passes.fake_tensor_prop.FakeTensorProp
    The only difference is that ValidationFakeProp supports int/float/bool in
    node.meta["val"]

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        mode: Optional[fake_tensor.FakeTensorMode] = None,
    ):
        super().__init__(module)
        if mode is None:
            mode = fake_tensor.FakeTensorMode()
        self._mode = mode

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)

        def extract_val(obj):
            if isinstance(obj, fake_tensor.FakeTensor):
                return proxy_tensor.snapshot_fake(obj)
            if isinstance(obj, torch.Tensor):
                return proxy_tensor.snapshot_fake(self._mode.from_tensor(obj))
            if isinstance(obj, proxy_tensor.py_sym_types):
                return obj
            if isinstance(obj, (int, float, bool)):
                # NOTE: These types are propagated into fx.Node of
                # (SymInt, SymFloat, SymBool)
                return obj
            return None

        meta = map_aggregate(result, extract_val)
        if meta is not None:
            n.meta["val"] = meta
        return result

    def propagate(self, *args):
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    def propagate_dont_convert_inputs(self, *args):
        with self._mode:
            return super().run(*args)
