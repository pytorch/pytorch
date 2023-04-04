from __future__ import annotations

import itertools

import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.passes import fake_tensor_prop
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
            fake_tensor_prop.FakeTensorProp(module, fake_tensor_mode).propagate(*args)

        return module
