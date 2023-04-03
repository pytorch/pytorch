from torch.ao.quantization.backend._qnnpack_pt2e import get_qnnpack_pt2e_backend_config

from .quantizer import Quantizer

import typing
import copy
from dataclasses import asdict, dataclass
from typing import Callable, Tuple, Union

import torch

SUPPORTED_DTYPES = [torch.uint8, torch.int8, torch.int32]
SUPPORTED_QSCHEMES = [
    torch.per_tensor_affine,
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
]


@dataclass
class TensorSpec:
    dtype: torch.dtype
    quant_min: int
    quant_max: int
    qscheme: torch.per_tensor_affine
    ch_axis: int
    is_dynamic: bool = False

    def __post_init__(self):
        # check dtype is one of the supported types
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype {self.dtype}.")

        # quant_min must be less than quant_max
        if self.quant_min > self.quant_max:
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # check qscheme is on of the supported ones
        if self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f"Unsupported qscheme {self.qscheme}.")

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


OperatorSpec = typing.NamedTuple(
    "OperatorSpec", [("activation", TensorSpec), ("weight", TensorSpec), ("bias", TensorSpec)]
)

SpecAndOperators = typing.NamedTuple(
    "SupportedSpecAndOperators",
    ("operator_spec", OperatorSpec),
    ("operators", Union[torch.nn.Module, Callable]),
)


def supported_symmetric_quantized_operators():
    supported_module = [torch.nn.Conv2d, torch.nn.functional.conv2d]
    return supported_module

def get_supported_symmetric_quantized_spec_and_operators() -> List[SpecAndOperators]:
    supported_spec_and_operators: List[SpecAndOperators] = []
    act_tensor_spec = TensorSpec(
        dtype=torch.qint8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_tensor_spec = None
    for qscheme in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]:
        weight_tensor_spec = TensorSpec(
            dtype=torch.qint8,
            quant_min=-127,
            quant_max=127,
            qscheme=qscheme,
            ch_axis=1,
            is_dynamic=False,
        )
        operator_spec = OperatorSpec(act_tensor_spec, weight_tensor_spec, bias_tensor_spec)
        for m in supported_symmetric_quantized_operators():
            supported_spec_and_operators.append((operator_spec, m))
    return copy.deepcopy(supported_spec_and_operators)

def get_supported_spec_and_operators() -> List[SpecAndOperators]:
    return get_supported_symmetric_quantized_spec_and_operators

class OperatorSpecConfig():

    def __init__(self):
        super().__init__()
        self.global_spec = None
        self.operator_type_specs = {}

    def set_global(self, operator_sepc: OperatorSpec) -> OperatorSpecConfig:
        self.global_spec = operator_spec
        return self

    def set_operator_type(self, obct_type: str, operator_spec: OperatorSpec) -> OperatorSpecConfig:
        self.operator_type_specs[operator_type] = operator_spec
        return self

class QNNPackQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.supported_spec_and_operators = get_supported_spec_and_operators()
        self.spec_config = SpecConfig()

    @classmethod
    def get_supported_operator_specs(self) -> List[OperatorSpec]:
        op_specs: Set[OperatorSpec] = {}
        for spec, operator in self.supported_spec_and_operators:
            op_specs.add(spec)
        return op_specs

    @classmethod
    def get_supported_operator_for_operator_spec(self, operator_spec: OperatorSpec) -> List[str]:
        ops: Set[str] = {}
        for spec, op in self.supported_spec_and_operators:
            if spec == operator_spec:
                ops.add(op)
        return ops

    def set_global(self, operator_spec: OperatorSpec):
        self.spec_config.set_global(operator_spec)

    def set_spec_for_operator_type(
        self, operator_type: str, operator_spec: OperatorSpec
    ):
        self.spec_config.set_operator_type(operator_type, operator_spec)

    def annotate(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    def validate(self, graph_module: torch.fx.GraphModule) -> None:
        pass
