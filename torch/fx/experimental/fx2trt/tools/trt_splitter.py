from typing import Dict, Iterable, Tuple, Mapping

import torch
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.experimental.fx2trt.tools.trt_minimizer import TensorRTMinimizer
from torch.fx.experimental.fx2trt import (
    InputTensorSpec,
    TRTModule,
    TRTInterpreter,
    CONVERTERS,
    NO_EXPLICIT_BATCH_DIM_SUPPORT,
    NO_IMPLICIT_BATCH_DIM_SUPPORT,
)
import torch.fx.passes.operator_support as ops
from torch.fx.passes.tools_common import Tensors
import torch.fx.experimental.fx_acc.acc_ops as acc_ops


class SkipOp:
    @classmethod
    def skip_op(cls, opsb: ops.OperatorSupportBase) -> ops.OperatorSupportBase:
        def _skip_op(
            submodules: Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            skipped_ops = [
                ("call_function", acc_ops.quantize_per_tensor)
            ]
            for op, target in skipped_ops:
                if node.op == op and node.target == target:
                    return True
            return opsb.is_node_supported(submodules, node)
        return ops.create_op_support(_skip_op)

def create_trt_operator_support(use_implicit_batch_dim=True) -> ops.OperatorSupportBase:
    """Creates an `OperatorSupportBase` instance used for TRT splitting purpose.
    """
    # Create an `OperatorSupport` that declares a node supported if it
    # finds a registered TRT converter.
    support_dict: Dict[str, None] = {}
    for k in CONVERTERS.keys():
        if use_implicit_batch_dim:
            if k not in NO_IMPLICIT_BATCH_DIM_SUPPORT.keys():
                support_dict[get_acc_ops_name(k)] = None
        elif k not in NO_EXPLICIT_BATCH_DIM_SUPPORT.keys():
            support_dict[get_acc_ops_name(k)] = None
    supported_if_converter_registered = ops.OperatorSupport(
        support_dict=support_dict
    )

    supported = ops.chain(
        # 1. Node is not supported if it has args with int64 dtype:
        SkipOp.skip_op(ops.OpSupports.decline_if_input_dtype(torch.int64)),
        # 2. Node is supported if it has TRT converter:
        supported_if_converter_registered,
    )
    return supported


class TRTSplitter(splitter_base._SplitterBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tuple[torch.Tensor],
        operator_support: ops.OperatorSupportBase = None,
        settings: splitter_base._SplitterSettingBase = None,
    ):
        if not operator_support:
            operator_support = create_trt_operator_support()
        if not settings:
            settings = splitter_base._SplitterSettingBase()
        super().__init__(module, sample_input, operator_support, settings, non_acc_submodule_name="_run_on_gpu_")

    def _lower_model_to_backend(
        self,
        mod: torch.fx.GraphModule,
        inputs: Iterable[torch.Tensor]
    ):
        """
        Lower a GraphModule `mod` to TensorRT with `inputs`.
        """
        # Current code for lowering is place-holder, subject to future change
        # based on feeds model's actual status
        interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
        interpreter_result = interp.run(*inputs)
        return TRTModule(interpreter_result.engine, interpreter_result.input_names, interpreter_result.output_names)

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors):
        """
        This function serves the preview functionality in Splitter. When previewing
        splitting result, if something wrong happens during lowering model to TensorRT
        or running a TensorRT model, this function will be called to find any culprit
        that is responsible for the error.
        """
        # Since we don't care about accuracy here, we pass in a dummy compare function.
        minimizer = TensorRTMinimizer(mod, inputs, lambda a, b, c: (1, True))
        minimizer.settings.traverse_method = "sequential"
        minimizer.settings.find_all = True
        culprits = minimizer.minimize()

        if len(culprits) == 0:
            reports = "Unable to find a culprit!\n"
        else:
            reports = "Found some problematic nodes:\n"
            for node in culprits:
                reports += f"{node.format_node()}\n"

        return reports


def get_acc_ops_name(k):
    if isinstance(k, str):
        return k
    elif k.__module__ and "acc_ops" in k.__module__:
        return f"acc_ops.{k.__name__}"
    else:
        module = k.__module__
        return f"{module if module else ''}.{k.__name__}".replace('_', '')
