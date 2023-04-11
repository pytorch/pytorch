from __future__ import annotations
from .quantizer import Quantizer

import copy
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Set, Dict, Callable
from torch.ao.quantization.observer import (
    PlaceholderObserver,
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
import torch
import operator
from torch.fx import Node

__all__ = [
    "QNNPackQuantizer",
    "get_default_symmetric_qnnpack_operator_qspec",
    "get_default_per_channel_symmetric_qnnpack_operator_qspec",
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

@dataclass(eq=True, frozen=True)
class QuantizationSpec:
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


OperatorQuantizationSpec = NamedTuple(
    "OperatorQuantizationSpec", [("activation", QuantizationSpec), ("weight", QuantizationSpec), ("bias", QuantizationSpec)]
)

OperatorQSpecAndOperators = NamedTuple(
    "OperatorQSpecAndOperators",
    [("operator_qspec", OperatorQuantizationSpec), ("operators", List[str])]
)


def supported_symmetric_quantized_operators() -> List[str]:
    supported_operators = ["conv2d", "linear", "add", "maxpool2d", "hardtanh", "mean", "adaptive_avgpool2d"]
    return copy.deepcopy(supported_operators)

def get_supported_symmetric_quantized_spec_and_operators() -> List[OperatorQSpecAndOperators]:
    supported_spec_and_operators: List[OperatorQSpecAndOperators] = []
    for operator_qspec in [get_default_symmetric_qnnpack_operator_qspec(), get_default_per_channel_symmetric_qnnpack_operator_qspec()]:
        ops = supported_symmetric_quantized_operators()
        supported_spec_and_operators.append(OperatorQSpecAndOperators(operator_qspec, ops))
    return copy.deepcopy(supported_spec_and_operators)

def get_default_symmetric_qnnpack_operator_qspec():
    act_qspec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_qspec = QuantizationSpec(dtype=torch.float)
    operator_qspec = OperatorQuantizationSpec(act_qspec, weight_qspec, bias_qspec)
    return operator_qspec

def get_default_per_channel_symmetric_qnnpack_operator_qspec():
    act_qspec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_qspec = QuantizationSpec(dtype=torch.float)
    operator_qspec = OperatorQuantizationSpec(act_qspec, weight_qspec, bias_qspec)
    return operator_qspec

def get_supported_spec_and_operators() -> List[OperatorQSpecAndOperators]:
    return get_supported_symmetric_quantized_spec_and_operators()

class OperatorQuantizationSpecConfig:

    def __init__(self):
        super().__init__()
        self.global_spec: Optional[OperatorQuantizationSpec] = None
        self.operator_type_specs: Dict[str, Optional[OperatorQuantizationSpec]] = {}

    def set_global(self, operator_qspec: Optional[OperatorQuantizationSpec]) -> OperatorQuantizationSpecConfig:
        self.global_spec = operator_qspec
        return self

    def set_operator_type(self, operator_type: str, operator_qspec: Optional[OperatorQuantizationSpec]) -> OperatorQuantizationSpecConfig:
        self.operator_type_specs[operator_type] = operator_qspec
        return self

# TODO: add support for torch dtype in quant code base
# this includes observers and prepare/convert code
_TORCH_DTYPE_TO_QDTYPE = {
    torch.int8: torch.qint8,
    torch.uint8: torch.quint8,
    torch.int32: torch.qint32,
    torch.float16: torch.float16,
}
def _get_act_obs_or_fq_ctr(operator_qspec: Optional[OperatorQuantizationSpec]):
    if operator_qspec is None:
        return None
    assert operator_qspec is not None
    qspec: QuantizationSpec = operator_qspec.activation
    qdtype = _TORCH_DTYPE_TO_QDTYPE[qspec.dtype]
    assert qspec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]
    if not qspec.is_dynamic:
        return HistogramObserver.with_args(
            dtype=qdtype,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
            reduce_range=False,
            eps=2**-12
        )
    else:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception("Unsupported quantization spec for activation: {}".format(qspec))

def _get_weight_obs_or_fq_ctr(operator_qspec: Optional[OperatorQuantizationSpec]):
    if operator_qspec is None:
        return None
    assert operator_qspec is not None
    qspec: QuantizationSpec = operator_qspec.weight
    qdtype = _TORCH_DTYPE_TO_QDTYPE[qspec.dtype]
    if qspec.qscheme == torch.per_tensor_symmetric:
        return MinMaxObserver.with_args(
            qscheme=qspec.qscheme,
            dtype=qdtype,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
            eps=2**-12
        )
    elif qspec.qscheme == torch.per_channel_symmetric:
        return PerChannelMinMaxObserver.with_args(
            qscheme=qspec.qscheme,
            dtype=qdtype,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
            eps=2**-12
        )
    else:
        raise Exception("Unsupported quantization spec for weight: {}".format(qspec))

def _get_bias_obs_or_fq_ctr(operator_qspec: Optional[OperatorQuantizationSpec]):
    if operator_qspec is None:
        return None
    assert operator_qspec is not None
    qspec: QuantizationSpec = operator_qspec.bias
    assert qspec.dtype == torch.float, "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=qspec.dtype)

def _get_default_obs_or_fq_ctr():
    return PlaceholderObserver.with_args(dtype=torch.float)

def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or ("target_dtype_info" in node.meta and node.meta["target_dtype_info"].get("_annotated", False))
    return annotated

class QNNPackQuantizer(Quantizer):
    supported_spec_and_operators = get_supported_spec_and_operators()

    def __init__(self):
        super().__init__()
        self.operator_qspec_config = OperatorQuantizationSpecConfig()

    @classmethod
    def get_supported_operator_qspecs(cls) -> List[OperatorQuantizationSpec]:
        op_specs: Set[OperatorQuantizationSpec] = set({})
        for spec, _ in cls.supported_spec_and_operators:
            op_specs.add(spec)
        return list(op_specs)

    @classmethod
    def get_supported_operator_for_operator_qspec(cls, operator_qspec: Optional[OperatorQuantizationSpec]) -> List[str]:
        if operator_qspec is None:
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
            if spec == operator_qspec:
                return ops
        return []

    def set_global(self, operator_qspec: Optional[OperatorQuantizationSpec]) -> QNNPackQuantizer:
        self.operator_qspec_config.set_global(operator_qspec)
        return self

    def set_spec_for_operator_type(
        self, operator_type: str, operator_qspec: Optional[OperatorQuantizationSpec]
    ) -> QNNPackQuantizer:
        self.operator_qspec_config.set_operator_type(operator_type, operator_qspec)
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

        global_spec = self.operator_qspec_config.global_spec
        ops = self.get_supported_operator_for_operator_qspec(global_spec)
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        for node in reversed(model.graph.nodes):
            for op in ops:
                if op == "conv2d":
                    self._annotate_conv2d_relu(node, global_spec)
                    self._annotate_conv2d(node, global_spec)
                elif op == "linear":
                    self._annotate_linear(node, global_spec)
                elif op == "maxpool2d":
                    self._annotate_maxpool2d(node, global_spec)
                elif op == "add":
                    self._annotate_add_relu(node, global_spec)
                    self._annotate_add(node, global_spec)
                elif op == "hardtanh":
                    self._annotate_hardtanh(node, global_spec)
                elif op == "mean":
                    self._annotate_mean(node, global_spec)
                elif op == "adaptive_avgpool2d":
                    self._annotate_adaptive_avg_pool2d(node, global_spec)

        return model

    def _annotate_conv2d_relu(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        if node.op != "call_function" or node.target not in [torch.ops.aten.relu_.default, torch.ops.aten.relu.default]:
            return
        relu_node = node
        conv_node = relu_node.args[0]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        if _is_annotated([relu_node, conv_node]):
            return

        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(operator_qspec),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }
        relu_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "_annotated": True,
        }

    def _annotate_conv2d(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        conv_node = node
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            return
        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(operator_qspec),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }

    def _annotate_linear(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        addmm_node = node
        if addmm_node.op != "call_function" or addmm_node.target != torch.ops.aten.addmm.default:
            return
        view_node = addmm_node.args[1]
        assert isinstance(view_node, Node)
        if view_node.op != "call_function" or view_node.target not in [torch.ops.aten.view.default, torch.ops.aten.view_copy.default]:
            return
        t_node = addmm_node.args[2]
        assert isinstance(t_node, Node)
        if t_node.op != "call_function" or t_node.target != torch.ops.aten.t.default:
            return
        if _is_annotated([addmm_node, view_node, t_node]):
            return

        # bias and output act
        addmm_node.meta["target_dtype_info"] = {
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(operator_qspec),
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "bias_index": 0,
            "_annotated": True,
        }
        # input act
        view_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        # weight
        t_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }

    def _annotate_maxpool2d(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        if node.op != "call_function" or node.target != operator.getitem or node.args[1] != 0:
            return
        getitem_node = node
        maxpool_node = getitem_node.args[0]
        assert isinstance(maxpool_node, Node)
        if maxpool_node.op != "call_function" or maxpool_node.target != torch.ops.aten.max_pool2d_with_indices.default:
            return
        if _is_annotated([getitem_node, maxpool_node]):
            return

        maxpool_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        getitem_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "input_output_share_observers": True,
            "_annotated": True,
        }

    def _annotate_input_out_obs_sharing_op(self, op: Callable, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        io_obs_sharing_node = node
        if io_obs_sharing_node.op != "call_function" or io_obs_sharing_node.target != op:
            return
        if _is_annotated([io_obs_sharing_node]):
            return

        io_obs_sharing_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "input_output_share_observers": True,
            "_annotated": True,
        }

    def _annotate_hardtanh(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        self._annotate_input_out_obs_sharing_op(torch.ops.aten.hardtanh.default, node, operator_qspec)

    def _annotate_mean(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        self._annotate_input_out_obs_sharing_op(torch.ops.aten.mean.default, node, operator_qspec)
        self._annotate_input_out_obs_sharing_op(torch.ops.aten.mean.dim, node, operator_qspec)

    def _annotate_adaptive_avg_pool2d(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        self._annotate_input_out_obs_sharing_op(torch.ops.aten.adaptive_avg_pool2d.default, node, operator_qspec)

    def _annotate_add_relu(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        if node.op != "call_function" or node.target not in [torch.ops.aten.relu_.default, torch.ops.aten.relu.default]:
            return
        relu_node = node
        add_node = relu_node.args[0]
        assert isinstance(add_node, Node)
        if add_node.op != "call_function" or add_node.target not in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
            return
        if _is_annotated([relu_node, add_node]):
            return

        add_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        relu_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "_annotated": True,
        }

    def _annotate_add(self, node: Node, operator_qspec: Optional[OperatorQuantizationSpec]) -> None:
        add_node = node
        if add_node.op != "call_function" or add_node.target not in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
            return
        if _is_annotated([add_node]):
            return

        add_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(operator_qspec),
            "_annotated": True,
        }

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
