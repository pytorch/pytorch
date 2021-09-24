from typing import Iterable, Tuple

import torch
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.experimental.fx2trt.tools.trt_minimizer import TensorRTMinimizer
from torch.fx.experimental.fx2trt.fx2trt import (
    InputTensorSpec,
    TRTModule,
    TRTInterpreter,
    CONVERTERS,
)
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import Tensors


class TRTOperatorSupport(OperatorSupport):
    def __init__(self):
        self._support_dict = {}
        for k in CONVERTERS.keys():
            name = self.get_op_name(k)
            self._support_dict[name] = None

    def get_op_name(self, k):
        if isinstance(k, str):
            return k
        elif k.__module__ and "acc_ops" in k.__module__:
            return f"acc_ops.{k.__name__}"
        else:
            module = k.__module__
            return f"{module if module else ''}.{k.__name__}".replace('_', '')




class TRTSplitter(splitter_base._SplitterBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tuple[torch.Tensor],
        operator_support: OperatorSupport = None,
        settings: splitter_base._SplitterSettingBase = None,
    ):
        if not operator_support:
            operator_support = TRTOperatorSupport()
        if not settings:
            settings = splitter_base._SplitterSettingBase()
        super().__init__(module, sample_input, operator_support, settings)

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
        engine, input_names, output_names = interp.run(*inputs)
        return TRTModule(engine, input_names, output_names)

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
