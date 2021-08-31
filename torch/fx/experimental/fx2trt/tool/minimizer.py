from typing import Tuple, Callable, Any

import torch
import torch.fx.passes.net_min_base as net_min_base
from deeplearning.trt.fx2trt.acc2trt import AcceleratorTRTInterpreter
from torch.fx.experimental.fx2trt.fx2trt import TRTModule, InputTensorSpec

'''
TensorRTMinimizer can help debug TensorRT errors during model lowering.
Example usage:
1. prepare model as mod
2. create minimizer:
    minimizer = TensorRTMinimizer(mod, your_inputs, compare_fn)
3. execute and debug:
    minimizer.run_nodes(start, end)
    run_nodes() can execute on all nodes if no start and end is specified;
    specifify different start&end can help narrow down problematic section of model;
    if not sure where to start debug, can start form use get_nodes() to get full
    picture of nodes.
'''
def lower_mod(mod: torch.fx.GraphModule, inputs: Tuple[torch.Tensor], batch_size=2048):
    interp = AcceleratorTRTInterpreter(mod, InputTensorSpec.from_tensors(inputs), explicit_batch_dimension=True)
    mod = TRTModule(*interp.run(max_batch_size=batch_size))
    return mod

class TensorRTMinizerSetting(net_min_base._MinimizerSettingBase):
    def __init__(self, max_batch_size=2048, explicit_batch_dimension=True):
        self.max_batch_size = max_batch_size
        self.explicit_batch_dimension = explicit_batch_dimension
        super.__init__()

class TensorRTMinimizer(net_min_base._MinimizerBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tuple[torch.Tensor],
        compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]],
        settings: TensorRTMinizerSetting = net_min_base._MinimizerSettingBase(),
    ):
        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(
        self,
        mod,
        inputs,
        lower_mod_fb: Callable[torch.fx.GraphModule, torch.Tensor], TRTModule=lower_mod
    ):
        mod.eval()
        try:
            mod = lower_mod_fb(mod, inputs, self.max_batch_size)
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}"
            )
        else:
            return output

    def get_nodes(self, start=None, end=None, enable_print=False):
        nodes = self._collect_nodes(start, end)
        if enable_print:
            print(f"Nodes fetched from start {start} to end {end} as: {nodes}")
        return nodes
