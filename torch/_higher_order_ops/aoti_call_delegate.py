# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


AOTI_LOWERED_MODULE = "AOTInductorEPModule"


class AOTICallDelegate(HigherOrderOperator):
    """aoti_call_delegate is a HOP for calling AOTInductor lowered submodule in ExportedProgram.

    It has the following signature:
    aoti_call_delegate(
        lowered_module: AOTInductorEPModule,
        original_gm:fx.GraphModule,
        weight_args: List[Tensor],
        input_args: List[Tensor],
    ) -> outputs: List[Tensor]

    where,
    - lowered_module is the AOTInductor lowered submodule, backed by compiled .so file, supporting real tensor inputs
    - original_gm is the original GraphModule before lowering, allowing FakeTensor propagation
    - weight_args is the list of weights in original GraphModule, including parameters and buffers
    - input_args is the list of flatten inputs

    NOTE: aoti_call_delegate doesn't support retracing yet, as original_gm is currently stateful with weight as get_attr nodes.
    This will fail functionalization during retrace. When we move AOTI to accept stateless GraphModule, we can enable retracing.

    When serialization, we have special hanlding for aoti_call_delegate, as AOTInductorEPModule is not serializable
    and stateful original_gm is failing the verifier.
    """

    def __init__(self) -> None:
        super().__init__("aoti_call_delegate")

    def __call__(
        self,
        lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
        original_gm: torch.fx.GraphModule,
        weight_args: list[torch.Tensor],
        input_args: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return super().__call__(lowered_module, original_gm, weight_args, input_args)


aoti_call_delegate = AOTICallDelegate()
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)


@aoti_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def call_delegate_cpu(
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    # FX creates this immutable_dict/list concept. Get rid of this.
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        input_args,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )

    has_fake_input_args = any(isinstance(arg, FakeTensor) for arg in new_args)
    has_fake_params = any(
        isinstance(param, FakeTensor) for param in original_gm.parameters()
    )
    has_fake_buffers = any(
        isinstance(buffer, FakeTensor) for buffer in original_gm.buffers()
    )

    if has_fake_input_args or has_fake_params or has_fake_buffers:
        # aoti lowered module doesn't support fake tensor
        return original_gm(*new_args)
    else:
        return lowered_module(new_args)  # type: ignore[misc]


@aoti_call_delegate.py_impl(FakeTensorMode)
# pyre-ignore
def call_delegate_fake_tensor_mode(
    mode: FakeTensorMode,
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    with mode:
        return call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)
