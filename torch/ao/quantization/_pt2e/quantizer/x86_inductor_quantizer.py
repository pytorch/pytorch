import torch
import copy
from .quantizer import Quantizer
from .qnnpack_quantizer import (
    _TORCH_DTYPE_TO_QDTYPE,
    _is_annotated,
    _get_default_obs_or_fq_ctr,
    SpecAndOperators,
    OperatorSpecConfig,
    TensorSpec,
    OperatorSpec,
)
from typing import List, Optional
from torch.fx import Node
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_operator_spec",
]

def supported_quantized_operators() -> List[str]:
    supported_operators = ["conv2d", ]
    return copy.deepcopy(supported_operators)

def _get_act_obs_or_fq_ctr(operator_spec: Optional[OperatorSpec]):
    if operator_spec is None:
        return None
    assert operator_spec is not None
    tensor_spec: TensorSpec = operator_spec.activation
    qdtype = _TORCH_DTYPE_TO_QDTYPE[tensor_spec.dtype]
    assert tensor_spec.qscheme in [torch.per_tensor_affine]
    if not tensor_spec.is_dynamic:
        return HistogramObserver.with_args(
            dtype=qdtype,
            quant_min=tensor_spec.quant_min,
            quant_max=tensor_spec.quant_max,
            reduce_range=False,
        )
    else:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception("Unsupported tensor_spec for activation: {}".format(tensor_spec))

def _get_weight_obs_or_fq_ctr(operator_spec: Optional[OperatorSpec]):
    if operator_spec is None:
        return None
    assert operator_spec is not None
    tensor_spec: TensorSpec = operator_spec.weight
    qdtype = _TORCH_DTYPE_TO_QDTYPE[tensor_spec.dtype]
    if tensor_spec.qscheme == torch.per_channel_symmetric:
        return PerChannelMinMaxObserver.with_args(
            qscheme=tensor_spec.qscheme,
            dtype=qdtype,
            quant_min=tensor_spec.quant_min,
            quant_max=tensor_spec.quant_max,
        )
    else:
        raise Exception("Unsupported tensor_spec for weight: {}".format(tensor_spec))

def _get_bias_obs_or_fq_ctr(operator_spec: Optional[OperatorSpec]):
    if operator_spec is None:
        return None
    assert operator_spec is not None
    tensor_spec: TensorSpec = operator_spec.bias
    assert tensor_spec.dtype == torch.float, "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=tensor_spec.dtype)

def get_default_x86_inductor_operator_spec():
    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_tensor_spec = TensorSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_tensor_spec = TensorSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 corresponding to weight shape = (oc, ic, kh, kw) of conv
        is_dynamic=False,
    )
    bias_tensor_spec = TensorSpec(dtype=torch.float)
    operator_spec = OperatorSpec(act_tensor_spec, weight_tensor_spec, bias_tensor_spec)
    return operator_spec

def get_supported_x86_inductor_quantized_spec_and_operators() -> List[SpecAndOperators]:
    supported_spec_and_operators: List[SpecAndOperators] = []
    for operator_spec in [get_default_x86_inductor_operator_spec(), ]:
        ops = supported_quantized_operators()
        supported_spec_and_operators.append(SpecAndOperators(operator_spec, ops))
    return copy.deepcopy(supported_spec_and_operators)

def get_supported_spec_and_operators() -> List[SpecAndOperators]:
    return get_supported_x86_inductor_quantized_spec_and_operators()

class X86InductorQuantizer(Quantizer):
    supported_spec_and_operators = get_supported_spec_and_operators()

    def __init__(self):
        super().__init__()
        self.operator_spec_config = OperatorSpecConfig()

    @classmethod
    def get_supported_operator_for_operator_spec(cls, operator_spec: Optional[OperatorSpec]) -> List[str]:
        if operator_spec is None:
            all_ops = []
            for _, ops in cls.supported_spec_and_operators:
                all_ops.extend(ops)
            return all_ops

        for spec, ops in cls.supported_spec_and_operators:
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if spec == operator_spec:
                return ops
        return []

    def set_global(self, operator_spec: Optional[OperatorSpec]):
        self.operator_spec_config.set_global(operator_spec)
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """ just handling global spec for now
        """
        # initialize default target_dtype_info
        _DEFAULT_TARGET_DTYPE_INFO = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
        }
        for node in model.graph.nodes:
            node.meta["target_dtype_info"] = copy.deepcopy(_DEFAULT_TARGET_DTYPE_INFO)

        global_spec = self.operator_spec_config.global_spec
        ops = self.get_supported_operator_for_operator_spec(global_spec)
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        for node in reversed(model.graph.nodes):
            for op in ops:
                if op == "conv2d":
                    self._annotate_conv2d(node, global_spec)
        return model

    def _annotate_conv2d(self, node: Node, operator_spec: Optional[OperatorSpec]) -> None:
        conv_node = node
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            return
        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_spec),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(operator_spec),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(operator_spec),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_spec),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
