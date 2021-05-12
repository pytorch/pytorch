from typing import Tuple, Dict, Callable, Any

import torch
import torch.fx
import torchvision.models as models
import torch.fx.experimental.fx2trt.converter.vanilla_converter
import torch.fx.passes.splitter_base as splitter_base
import torch.fx.passes.operator_support as op_support
import torch.fx.passes.net_min_base as net_min_base
from torch.fx.experimental.fx2trt.fx2trt import TRTInterpreter, InputTensorSpec, TRTModule


# The purpose of this example is to demonstrate the overall flow of lowering a PyTorch
# model to TensorRT via FX with existing FX based tooling. The general lowering flow
# would be like:
#
# 1. Use splitter to split the model if there're ops in the model that we don't want to
#    lower to TensorRT for some reasons like the ops are not supported in TensorRT or
#    running them on other backends provides better performance.
# 2. Lower the model (or part of the model if splitter is used) to TensorRT via fx2trt.
#
# For this example, we use ResNet18 as example model and split out the linear layer to
# not run on TensorRT just to demonstrate how the splitter works. At the end of this
# example we did a benchmark for a model (named `split_mod`) with all the ops running
# on TensorRT execpt linear layer running on PyTorch Cuda versus a model (named `rn18`)
# fully on PyTorch Cuda.


# Create ResNet18 `rn18` and inputs `x`
rn18 = models.resnet18().eval().cuda()
x = torch.randn(5, 3, 224, 224, device="cuda")

# Trace the model with FX.
traced_rn18 = torch.fx.symbolic_trace(rn18)


def lower_mod_to_trt(mod: torch.fx.GraphModule, inputs: Tuple[torch.Tensor]):
    """
    Helper function that given a GraphModule `mod` and its `inputs`, build a
    TRTModule that runs the original `mod` on TensorRT.
    """
    interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
    engine, input_names, output_names = interp.run(*inputs)
    return TRTModule(engine, input_names, output_names)


class OpSupport(op_support.OperatorSupport):
    """
    This class is used by splitter to determine which nodes are supported, i.e.
    should be split to the accelerator part (TensorRT).
    """
    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ):
        """
        Here we want linear layer to not run on TensorRT. Thus, we return
        False for linear layer and True for all other ops.
        """
        target = op_support.get_node_target(submodules, node)

        if target == "torch.nn.modules.linear.Linear":
            return False

        return True


class TensorRTMinimizer(net_min_base._MinimizerBase):
    """
    Need to define a Minimizer class for TensorRT because it's used in Splitter.
    """
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tuple[torch.Tensor],
        compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]],
        settings: net_min_base._MinimizerSettingBase = None,
    ):
        if settings is None:
            settings = net_min_base._MinimizerSettingBase()

        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        """
        The output of this function serves as an reference.
        """
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(self, mod, inputs):
        """
        Here we actually run mod on TensorRT return TensorRT result.
        """
        mod.eval()
        try:
            mod = lower_mod_to_trt(mod, inputs)
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}"
            )
        else:
            return output


# This in the future will be a global TensorRTSplitter and we don't need to create
# it per example.
class TensorRTSplitter(splitter_base._SplitterBase):
    """
    Splitter for TensorRT.
    """
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tuple[torch.Tensor],
        operator_support: op_support.OperatorSupport = None,
        settings: splitter_base._SplitterSettingBase = None
    ):
        if not operator_support:
            operator_support = op_support.OperatorSupport()

        if not settings:
            settings = splitter_base._SplitterSettingBase()
            settings.allow_non_tensor = True
            settings.skip_fusion = True

        super().__init__(module, sample_input, operator_support, settings)

    def _lower_model_to_backend(self, mod, inputs):
        """
        Lower a GraphModule `mod` to TensorRT with `inputs`.
        """
        mod = lower_mod_to_trt(mod, inputs)
        return mod

    def _find_culprit(self, mod, inputs):
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

# Create a splitter which takes in traced ResNet18.
splitter = TensorRTSplitter(traced_rn18, (x,), OpSupport())

# node_support_preview() shows the details of node supporting information based
# on the DummyOpSupport we created.
#
# In the output, we have supported node types
# and unsupported node types. Nodes in the model with supported types will be
# split into accelerator submodules while nodes with unsupported types will be
# split into cpu submodules.
splitter.node_support_preview()
"""
output:

Supported node types in the model:
torch.nn.modules.conv.Conv2d: ((torch.float32,), {})
torch.nn.modules.batchnorm.BatchNorm2d: ((torch.float32,), {})
torch.nn.modules.activation.ReLU: ((torch.float32,), {})
torch.nn.modules.pooling.MaxPool2d: ((torch.float32,), {})
_operator.add: ((torch.float32, torch.float32), {})
torch.nn.modules.pooling.AdaptiveAvgPool2d: ((torch.float32,), {})
torch.flatten: ((torch.float32,), {})

Unsupported node types in the model:
torch.nn.modules.linear.Linear: ((torch.float32,), {})
"""

# split_preview() shows the details of how the model looks like after split.
# And for every accelerator module in the split model, it would run a check
# by lowering and running the module. If any error is catched during the
# checking process, it will try to find which nodes are causing the trouble
# here with minimizer.
#
# Notice that after split, the model will have some submodules called either
# `_run_on_acc_{}` or `_run_on_cpu_{}`. We have all the supported nodes in
# `_run_on_acc_{}` modules and all other nodes in `_run_on_cpu_{}` modules.
#
# In the output, we can see it estimates the max qps based on PCIe bandwidth,
# this is something we need to consider when lowering to acceleartors chips,
# because the data will be flowing between cpu and accelerator which might not
# matter in GPU case.
splitter.split_preview()
"""
output:

Before removing small acc subgraphs, total 2 subgraphs are created: 1 acc subgraphs and 1 cpu subgraphs.
After removing small acc subgraphs, total 2 subgraphs are created: 1 acc subgraphs and 1 cpu subgraphs.
_run_on_acc_0: 68 node(s)
_run_on_cpu_1: 1 node(s)

Processing acc submodule _run_on_acc_0
Checking inputs...
Checking outputs...
Total input size in bytes is 3010560, total output size in bytes is 10240, theoretical max qps (bounds by PCIe bandwidth)
for this submodule is 35665.85034013606.
Lowering and running succeed!

Theoretical max qps (bounds by PCIe bandwidth) for this model is 35665.85034013606, bottleneck is submodule _run_on_acc_0.
"""

# After split we have two submodules, one is `_run_on_acc_0` and one is `_run_on_cpu_1`.
# We have only one op in `_run_on_cpu_1` which is a linear layer while all other ops are
# in `_run_on_acc_0`.
split_mod = splitter()
print(split_mod.graph)
"""
output:

graph():
    %x : torch.Tensor [#users=1] = placeholder[target=x]
    %_run_on_acc_0 : [#users=1] = call_module[target=_run_on_acc_0](args = (%x,), kwargs = {})
    %_run_on_cpu_1 : [#users=1] = call_module[target=_run_on_cpu_1](args = (%_run_on_acc_0,), kwargs = {})
    return _run_on_cpu_1
"""

# We want to lower _run_on_acc_0 to TensorRT.
split_mod._run_on_acc_0 = lower_mod_to_trt(split_mod._run_on_acc_0, (x,))  # type: ignore[arg-type]

# Assert results are equal with the original model.
rn18 = rn18.cuda()
torch.testing.assert_allclose(split_mod(x), rn18(x))

import time
NITER = 100

s = time.time()
for _ in range(NITER):
    split_mod(x)
    torch.cuda.synchronize()
print('trt time (ms/iter)', (time.time() - s) / NITER * 1000)
"""
output:

trt time (ms/iter) 1.978142261505127
"""

s = time.time()
for _ in range(NITER):
    rn18(x)
    torch.cuda.synchronize()
print('stock PyTorch time (ms/iter)', (time.time() - s) / NITER * 1000)
"""
output:

stock PyTorch time (ms/iter) 3.8208484649658203
"""
