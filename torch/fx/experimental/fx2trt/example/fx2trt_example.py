# type: ignore[]

import torch
import torch.fx
import torch.nn as nn
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.tools.trt_splitter import TRTSplitter
from torch.fx.experimental.fx2trt import TRTInterpreter, InputTensorSpec, TRTModule


# The purpose of this example is to demonstrate the overall flow of lowering a PyTorch
# model to TensorRT via FX with existing FX based tooling. The general lowering flow
# would be like:
#
# 1. Use splitter to split the model if there're ops in the model that we don't want to
#    lower to TensorRT for some reasons like the ops are not supported in TensorRT or
#    running them on other backends provides better performance.
# 2. Lower the model (or part of the model if splitter is used) to TensorRT via fx2trt.
#
# If we know the model is fully supported by fx2trt then we can skip the splitter.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.linalg.norm(x, ord=2, dim=1)
        return x

inputs = [torch.randn(1, 10)]
model = Model().eval()

# acc_tracer is a custom fx tracer that maps nodes whose targets are PyTorch operators
# to acc ops.
traced = acc_tracer.trace(model, inputs)

# Splitter will split the model into serveral submodules. The name of submodules will
# be either `run_on_acc_{}` or `run_on_gpu_{}`. Submodules named `run_on_acc_{}` can
# be fully lowered to TensorRT via fx2trt while submodules named `run_on_gpu_{}` has
# unsupported ops and can't be lowered by fx2trt. We can still run `run_on_gpu_{}`
# submodules on Gpu if ops there have cuda implementation, the naming is a bit
# confusing and we'll improve it.
splitter = TRTSplitter(traced, inputs)

# Preview functionality allows us to see what are the supported ops and unsupported
# ops. We can optionally the dot graph which will color supported ops and unsupported
# ops differently.
splitter.node_support_preview(dump_graph=False)
"""
Supported node types in the model:
acc_ops.linear: ((), {'input': torch.float32, 'weight': torch.float32, 'bias': torch.float32})
acc_ops.relu: ((), {'input': torch.float32})

Unsupported node types in the model:
acc_ops.linalg_norm: ((), {'input': torch.float32})
"""

# Split.
split_mod = splitter()

# After split we have two submodules, _run_on_acc_0 and _run_on_gpu_1.
print(split_mod.graph)
"""
graph():
    %x : [#users=1] = placeholder[target=x]
    %_run_on_acc_0 : [#users=1] = call_module[target=_run_on_acc_0](args = (%x,), kwargs = {})
    %_run_on_gpu_1 : [#users=1] = call_module[target=_run_on_gpu_1](args = (%_run_on_acc_0,), kwargs = {})
    return _run_on_gpu_1
"""

# Take a look at what inside each submodule. _run_on_acc_0 contains linear and relu while
# _run_on_gpu_1 contains linalg_norm which currently is not supported by fx2trt.
print(split_mod._run_on_acc_0.graph)
print(split_mod._run_on_gpu_1.graph)
"""
graph():
    %x : [#users=1] = placeholder[target=x]
    %linear_weight : [#users=1] = get_attr[target=linear.weight]
    %linear_bias : [#users=1] = get_attr[target=linear.bias]
    %linear_1 : [#users=1] = call_function[target=torch.fx.experimental.fx_acc.acc_ops.linear](args = (), ...
    %relu_1 : [#users=1] = call_function[target=torch.fx.experimental.fx_acc.acc_ops.relu](args = (), ...
    return relu_1
graph():
    %relu_1 : [#users=1] = placeholder[target=relu_1]
    %linalg_norm_1 : [#users=1] = call_function[target=torch.fx.experimental.fx_acc.acc_ops.linalg_norm](args = (), ...
    return linalg_norm_1
"""

# Now let's lower split_mod._run_on_acc_0. If we know the model can be fully lowered,
# we can skip the splitter part.
interp = TRTInterpreter(split_mod._run_on_acc_0, InputTensorSpec.from_tensors(inputs))
engine, input_names, output_names = interp.run()
trt_mod = TRTModule(engine, input_names, output_names)
split_mod._run_on_acc_0 = trt_mod

cuda_inputs = [input.cuda() for input in inputs]
split_mod.cuda()
lowered_model_output = split_mod(*cuda_inputs)

# Make sure the results match
model.cuda()
regular_model_output = model(*cuda_inputs)
torch.testing.assert_close(lowered_model_output, regular_model_output.to(torch.float16), atol=3e-3, rtol=1e-2)

# We can utilize the trt profiler to print out the time spend on each layer.
trt_mod.enable_profiling()
trt_mod(*cuda_inputs)
'''
Reformatting CopyNode for Input Tensor 0 to LayerType.FULLY_CONNECTED_acc_ops.linear_linear_1: 0.027392ms
LayerType.FULLY_CONNECTED_acc_ops.linear_linear_1: 0.023072ms
PWN(ActivationType.RELU_acc_ops.relu_relu_1): 0.008928ms
'''
trt_mod.disable_profiling()
