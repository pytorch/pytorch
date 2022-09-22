from enum import IntEnum
from typing import Any

import torch
import torch.fx
from torch.fx import subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import wrap_torch_function
from torch.utils._pytree import _register_pytree_node

# (1) - (3): the parts that are related to fp8 dtype
# 1 - 5: steps in flows

# (1). How is `fp8` dtype defined?
# It's defined as an out of core enum, as follows:
# we can define this in some qdtype.py file, so it's clear this is a dtype
class qdtype(IntEnum):
    FP8 = 0

# # we also need to define the quantize/dequantize functions for this dtype, or we can add new branches to existing implementations
@torch.fx.wrap
def quantize_per_tensor_affine_fp8(t: torch.Tensor, dtype: Any, scale: float):
    # not correct, just for demo
    if dtype == qdtype.FP8:
        return (t / scale).char()
    raise NotImplementedError


@torch.fx.wrap
def dequantize_per_tensor_affine_fp8(t: torch.Tensor, dtype: Any, scale: float):
    if dtype == qdtype.FP8:
        return t.float() * scale
    raise NotImplementedError


# NOTE: this can be removed if the subgraph_rewriter problem is resolved
class QTensor:
    # NB: wrap_torch_function so that this "factory" function can be
    # symbolically traced as is.  This is not strictly necessary but
    # makes constructing patterns more convenient.
    @staticmethod
    @wrap_torch_function(lambda t, x, y: (t,))
    def quantize(t: torch.Tensor, dtype: Any, scale: float):
        return quantize_per_tensor_affine_fp8(t, dtype, scale)

    # adding scale/zero_point paramters for easier pattern matching
    @wrap_torch_function(lambda t, x, y: (t,))
    def dequantize(t: torch.Tensor, dtype: Any, scale: float):
        return dequantize_per_tensor_affine_fp8(t, dtype, scale)



# Let's take a simple model that adds two Tensors

class M(torch.nn.Module):
    def forward(self, inp, add_weight):
        return inp + add_weight


# We use the pattern matching API to look for occurrences of add.

# We use make_fx to generate the sequence of ATen ops that correspond to a
# add call.  Note that this pattern is only valid if there aren't any
# conditions on, e.g., the shapes of the input tensor.  In general you
# may need a pattern for every permutation of how a composite operator may
# lower; you can get all of them by running through a sufficiently large
# number of example inputs.
# TODO: use symbolic shapes here; this would give you a series of guards
# that would tell you what input sizes the pattern is valid for.
# linear_pattern = make_fx(lambda i, w: torch.nn.functional.linear(i, w))(torch.randn(0, 0), torch.randn(0, 0))

# We first trace out the ATen OP IR of the original model
inp = torch.randn(4, 4)
weight = torch.randn(4, 4)
m = M().eval()
m = make_fx(m)(inp, weight)
print("original:", m)

# 1. Prepare for observation (insert observers)
# NOTE: we may need to define new observers for the new dtype, for now I'll just use the default one

# Insert observer objects
# with ad-hoc rules for now
from torch.ao.quantization import default_qconfig, is_activation_post_process


def insert_observer_for(m, node, obs_name):
    graph = m.graph
    named_modules = dict(m.named_modules())
    if node.op == "call_module" and is_activation_post_process(named_modules[node.target]):
        return
    with graph.inserting_after(node):
        observer_node = graph.create_node("call_module", obs_name, (node,), {})
        node.replace_all_uses_with(observer_node)
        observer_node.replace_input_with(observer_node, node)


counter = 0
for n in m.graph.nodes:
    if n.target == torch.ops.aten.add.Tensor:
        input_observer = default_qconfig.activation()
        obs_name = "obs_" + str(counter)
        m.register_module(obs_name, input_observer)
        counter += 1
        insert_observer_for(m, n.args[0], obs_name)

        input_observer = default_qconfig.activation()
        obs_name = "obs_" + str(counter)
        m.register_module(obs_name, input_observer)
        counter += 1
        insert_observer_for(m, n.args[1], obs_name)

        output_observer = default_qconfig.activation()
        obs_name = "obs_" + str(counter)
        m.register_module(obs_name, output_observer)
        counter += 1
        insert_observer_for(m, n, obs_name)

m.recompile()
print("1. observed:", m)

# 2. Calibration with example dataset
# calibration step for observer to learn about the statistics of the tensors they are observing
for _ in range(10):
    m(inp, weight)


# 3. Convert (to reference quantized model with out of core QTneosr)

named_modules = dict(m.named_modules())
counter = 0
# Replace observers with q/dq ops
for n in m.graph.nodes:
    if n.op == "call_module" and is_activation_post_process(named_modules[n.target]):
        observer = named_modules[n.target]
        scale, zero_point = observer.calculate_qparams()
        with m.graph.inserting_before(n):
            scale_name = "_scale_" + str(counter)
            m.register_buffer(scale_name, scale)
            scale_node = m.graph.create_node("get_attr", scale_name, (), {})
            # converting to int since only basic types are supported as args for fx Node
            quantize_node = m.graph.create_node(
                "call_function",
                QTensor.quantize,
                (n.args[0], int(qdtype.FP8), scale_node),
                {},
            )
            dequantize_node = m.graph.create_node(
                "call_function",
                QTensor.dequantize,
                (quantize_node, int(qdtype.FP8), scale_node),
                {},
            )
            n.replace_all_uses_with(dequantize_node)
            dequantize_node.replace_input_with(dequantize_node, quantize_node)
            m.graph.erase_node(n)
m.recompile()
print("3. reference quantized model with out of core QTensor:", m)


# (2). How are the operators that works with fp8 Tensors are defined?
# this is how the fp8 add should look like:
# basically `a` and `b` are normal int8 Tensors
# but `a` and `a_scale` can be interpreted as a fp8 Tensor by the operator
# NOTE: in the implementation we dequantize the tensors, but this just for demo, in reality we should implmeent fp8 arithmetic in some
# efficient libraries directly
@torch.fx.wrap
def per_tensor_affine_fp8_add(a, a_scale, b, b_scale, output_scale):
    # a and b are int8 Tensors, but can be interpreted as fp8 Tensors
    a_float = QTensor.dequantize(a, qdtype.FP8, a_scale)
    b_float = QTensor.dequantize(b, qdtype.FP8, b_scale)
    output = a_float + b_float
    output = QTensor.quantize(output, qdtype.FP8, output_scale)
    return output

# (2.1) Which opset (aten or prim) should this operator be defined?
# This depends on where do we want to do lowering, in the overall flow we can do lowering in Step 4 or 5
# if we do lowering in Step 4, and Step 4 is lowering to aten opset, then we'd need to define this op in aten op set and prim opset
# otherwise, we only need to define this in prim opset


# TODO: we need a filter to make sure `a_dtype` == `b_dtype` == FP8 in subgraph_rewriter
def add_pattern_fn(a, a_dtype, a_scale, b, b_dtype, b_scale, out_scale):
    # pattern with QTensor quantize and dequantize is much simpler than patterns with prim ops
    a = QTensor.dequantize(a, a_dtype, a_scale)
    b = QTensor.dequantize(b, b_dtype, b_scale)
    output = torch.ops.aten.add.Tensor(a, b)
    output = QTensor.quantize(output, a_dtype, out_scale)
    return output


def add_replace_fn(a, a_dtype, a_scale, b, b_dtype, b_scale, out_scale):
    return per_tensor_affine_fp8_add(a, a_scale, b, b_scale, out_scale)

# (3). How does lowering work for `fp8` operators? We can lower from the following two steps
# in Step 4. we lower from quantize/dequantize operators representation:
# e.g. dequant -> add -> quant
# in Step 5. we lower from the decomposed quantize/dequantize operator representation:
# e.g. t -> subtract -> mul -> add -> divide -> add
# where "subtract -> mul" is decomopsed dequantize and "divide -> add" is decomposed quantize


# gm = torch.fx.symbolic_trace(add_pattern_fn)
# gm.print_readable()
# exit()

# 4. Lowering to quantized operators that does not depend on quantized Tensors

# Now, we replace occurrences of add with quantize/dequantize
matches = subgraph_rewriter.replace_pattern(m, add_pattern_fn, add_replace_fn)
print(len(matches), "matches found")

print("4 (optional). after lowering (in pytorch) (quantized ops + q/dq ops):", m)
m(inp, weight)

# 5. Remove q/dq ops from the graph, quantized ops from previous step should survive the trace (Note in the example the quantized ops are also traced away)

# Finally, we will need to retrace the model to get lowered operations in terms
# of only pure PyTorch operations if we have to lower the model in a different environment
# TODO: use an interpreter here to preserve stack traces
m = make_fx(m)(inp, weight)
m(inp, weight)
print("5. quantized without q/dq ops for lowering in a different environment:", m)