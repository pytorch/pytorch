from __future__ import annotations
from .quantizer import Quantizer

import copy
from dataclasses import dataclass
from typing import Callable, Tuple, Union, List, NamedTuple, Optional
from torch.ao.quantization.observer import (
    PlaceholderObserver,
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.fx import Node

import torch

__all__ = [
    "QNNPackQuantizer",
    "get_default_symmetric_qnnpack_operator_spec",
    "get_default_per_channel_symmetric_qnnpack_operator_spec",
]

# TODO: maybe remove torch.float32
SUPPORTED_DTYPES = [torch.uint8, torch.int8, torch.int32, torch.float16, torch.float32]
SUPPORTED_QSCHEMES = [
    torch.per_tensor_affine,
    torch.per_tensor_symmetric,
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
]


# Note: do we need order=True? this is just added so that
# selecting an supported_operator_spec is easeir (returns the same thing every time)
# we can remove it later when we have easier apis for users to get an supported
# operator spec
@dataclass(eq=True, frozen=True)
class TensorSpec:
    dtype: torch.dtype
    is_dynamic: bool = False
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None

    def __post_init__(self):
        # check dtype is one of the supported types
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype {self.dtype}.")

        # quant_min must be less than quant_max
        if self.quant_min is not None and self.quant_max is not None and self.quant_min > self.quant_max:
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # check qscheme is on of the supported ones
        if self.qscheme is not None and self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f"Unsupported qscheme {self.qscheme}.")

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


OperatorSpec = NamedTuple(
    "OperatorSpec", [("activation", TensorSpec), ("weight", TensorSpec), ("bias", TensorSpec)]
)

SpecAndOperators = NamedTuple(
    "SupportedSpecAndOperators",
    [("operator_spec", OperatorSpec), ("operators", List[str])]
)


def supported_symmetric_quantized_operators():
    supported_operators = ["conv2d"]
    return copy.deepcopy(supported_operators)

def get_supported_symmetric_quantized_spec_and_operators() -> List[SpecAndOperators]:
    supported_spec_and_operators: List[SpecAndOperators] = []
    for operator_spec in [get_default_symmetric_qnnpack_operator_spec(), get_default_per_channel_symmetric_qnnpack_operator_spec()]:
        for ops in supported_symmetric_quantized_operators():
            supported_spec_and_operators.append((operator_spec, ops))
    return copy.deepcopy(supported_spec_and_operators)

def get_default_symmetric_qnnpack_operator_spec():
    act_tensor_spec = TensorSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_tensor_spec = TensorSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_tensor_spec = TensorSpec(dtype=torch.float)
    operator_spec = OperatorSpec(act_tensor_spec, weight_tensor_spec, bias_tensor_spec)
    return operator_spec

def get_default_per_channel_symmetric_qnnpack_operator_spec():
    act_tensor_spec = TensorSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_tensor_spec = TensorSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_tensor_spec = TensorSpec(dtype=torch.float)
    operator_spec = OperatorSpec(act_tensor_spec, weight_tensor_spec, bias_tensor_spec)
    return operator_spec

def get_supported_spec_and_operators() -> List[SpecAndOperators]:
    return get_supported_symmetric_quantized_spec_and_operators()

class OperatorSpecConfig:

    def __init__(self):
        super().__init__()
        self.global_spec = None
        self.operator_type_specs = {}

    def set_global(self, operator_spec: OperatorSpec) -> OperatorSpecConfig:
        self.global_spec = operator_spec
        return self

    def set_operator_type(self, operator_type: str, operator_spec: OperatorSpec) -> OperatorSpecConfig:
        self.operator_type_specs[operator_type] = operator_spec
        return self

# TODO: add support for torch dtype in quant code base
# this includes observers and prepare/convert code
_TORCH_DTYPE_TO_QDTYPE = {
    torch.int8: torch.qint8,
    torch.uint8: torch.quint8,
    torch.int32: torch.qint32,
    torch.float16: torch.float16,
}
def _get_act_observer(tensor_spec: TensorSpec):
    qdtype = _TORCH_DTYPE_TO_QDTYPE[tensor_spec.dtype]
    assert tensor_spec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]
    if not tensor_spec.is_dynamic:
        return HistogramObserver.with_args(dtype=qdtype, quant_min=tensor_spec.quant_min, quant_max=tensor_spec.quant_max, reduce_range=False, eps=2**-12)
    else:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception("Unsupported tensor_spec for activation: {}".format(tensor_spec))

def _get_weight_observer(tensor_spec: TensorSpec):
    qdtype = _TORCH_DTYPE_TO_QDTYPE[tensor_spec.dtype]
    if tensor_spec.qscheme == torch.per_tensor_symmetric:
        return MinMaxObserver.with_args(qscheme=tensor_spec.qscheme, dtype=qdtype, quant_min=tensor_spec.quant_min, quant_max=tensor_spec.quant_max, eps=2**-12)
    elif tensor_spec.qscheme == torch.per_channel_symmetric:
        return PerChannelMinMaxObserver.with_args(qscheme=tensor_spec.qscheme, dtype=qdtype, quant_min=tensor_spec.quant_min, quant_max=tensor_spec.quant_max, eps=2**-12)
    else:
        raise Exception("Unsupported tensor_spec for weight: {}".format(tensor_spec))

def _get_bias_observer(tensor_spec: TensorSpec):
    assert tensor_spec.dtype == torch.float, "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=tensor_spec.dtype)

class QNNPackQuantizer(Quantizer):
    supported_spec_and_operators = get_supported_spec_and_operators()

    def __init__(self):
        super().__init__()
        self.operator_spec_config = OperatorSpecConfig()

    @classmethod
    def get_supported_operator_specs(cls) -> List[OperatorSpec]:
        op_specs: Set[OperatorSpec] = set({})
        for spec, operator in cls.supported_spec_and_operators:
            op_specs.add(spec)
        return list(op_specs)

    @classmethod
    def get_supported_operator_for_operator_spec(self, operator_spec: OperatorSpec) -> List[str]:
        ops: Set[str] = set({})
        for spec, op in self.supported_spec_and_operators:
            if spec == operator_spec:
                ops.add(op)
        return ops

    def set_global(self, operator_spec: OperatorSpec):
        self.operator_spec_config.set_global(operator_spec)

    def set_spec_for_operator_type(
        self, operator_type: str, operator_spec: OperatorSpec
    ):
        self.operator_spec_config.set_operator_type(operator_type, operator_spec)

    def annotate(self, model: torch.fx.GraphModule) -> None:
        """ just handling global spec for now
        """
        global_spec = self.operator_spec_config.global_spec
        ops = self.get_supported_operator_for_operator_spec(global_spec)
        for op in ops:
            self._annotate_op(model, op, global_spec)

    def _annotate_op(self, model: torch.fx.GraphModule, op: str, operator_spec: OperatorSpec) -> None:
        _DEFAULT_TARGET_DTYPE_INFO = {
            "input_act_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=torch.float),
            "output_act_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=torch.float),
        }
        assert op == "conv2d", "only conv2d is supported right now"
        for node in model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.convolution.default:
                node.meta["target_dtype_info"] = {
                    "input_act_obs_or_fq_ctr": _get_act_observer(operator_spec.activation),
                    "weight_obs_or_fq_ctr": _get_weight_observer(operator_spec.weight),
                    "bias_obs_or_fq_ctr": _get_bias_observer(operator_spec.bias),
                    "output_act_obs_or_fq_ctr": _get_act_observer(operator_spec.activation),
                    # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
                    "weight_index": 1,
                    # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
                    "bias_index": 2,
                }
                # set default target_dtype_info for input args
                for arg in node.args:
                    if isinstance(arg, Node) and "target_dtype_info" not in arg.meta:
                        arg.meta["target_dtype_info"] = _DEFAULT_TARGET_DTYPE_INFO

    def validate(self, graph_module: torch.fx.GraphModule) -> None:
        pass
