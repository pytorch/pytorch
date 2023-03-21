from __future__ import annotations

import itertools

import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.passes import fake_tensor_prop
from torch.nn.utils import stateless

from torch.onnx._internal import _beartype


@_beartype.beartype
def shape_inference_with_fake_tensor(decomposed_module: torch.fx.GraphModule, *args):
    # Use this FakeTensorMode to
    # 1. convert nn.Parameter's in nn.Module to FakeTensor
    # 2. run FakeTensorProp
    # If (1) and (2) are done with difference FakeTensorMode's, undefined behavior may
    # happen.
    fake_tensor_mode = fake_tensor.FakeTensorMode()

    def to_fake_tensor(x):
        if isinstance(x, torch.Tensor) and not isinstance(x, fake_tensor.FakeTensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    # "args" are FakeTensor in FakeTensorProp so the parameters and buffers
    # in model must be converted to FakeTensor as well.
    fake_parameters_and_buffers = {
        k: to_fake_tensor(v)
        for k, v in itertools.chain(
            decomposed_module.named_parameters(), decomposed_module.named_buffers()
        )
    }

    # input nodes (plaveholder) can never be skipped in FakeTensorProp,
    # or made up inputs would be generated in the graph
    initial_env = {
        node: node.meta["val"]
        for node in decomposed_module.graph.nodes
        if "val" in node.meta and node.op != "placeholder"
    }

    # Shape inference via FakeTensorProp
    with stateless._reparametrize_module(
        decomposed_module, fake_parameters_and_buffers
    ):
        # Assign output types and shapes to each node without meta values.
        fake_tensor_prop.FakeTensorProp(decomposed_module, fake_tensor_mode).propagate(
            *args, initial_env=initial_env
        )

    return decomposed_module
