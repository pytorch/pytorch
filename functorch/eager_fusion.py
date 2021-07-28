import time
import torch
import torch.nn as nn
from functorch import make_fx, grad, nnc_jit, nnc_compile, vmap, make_nnc, vjpfull
from torch.fx.node import map_arg
import torch.fx as fx
from torchvision.models import resnet18

torch.manual_seed(0)
def partition_backwards(fx_module: fx.GraphModule):
    bw_nodes = set()
    saved_nodes = set()
    output_node = None
    for n in fx_module.graph.nodes:
        if n.op == 'placeholder' and 'tangents' in n.target:
            bw_nodes.add(n)
        elif n.op != 'output':
            has_color = False
            def is_colored(a):
                nonlocal has_color
                if a in bw_nodes or a in saved_nodes:
                    has_color = True
            def add_saved(a):
                if a not in bw_nodes:
                    saved_nodes.add(a)
            map_arg(n.args, lambda x: is_colored(x))
            map_arg(n.kwargs, lambda x: is_colored(x))
            if has_color:
                bw_nodes.add(n)
                map_arg(n.args, lambda x: add_saved(x))
                map_arg(n.kwargs, lambda x: add_saved(x))
        elif n.op == 'output':
            output_node = n

    bw_graph = fx.Graph()
    value_remap = {}
    for saved_node in saved_nodes:
        value_remap[saved_node] = bw_graph.placeholder(saved_node.name)

    for node in fx_module.graph.nodes:
        if node in bw_nodes:
            value_remap[node] = bw_graph.node_copy(node, lambda n : value_remap[n])
    num_fwd_outputs = fx_module._out_spec.children_specs[0].num_leaves
    num_bwd_outputs = fx_module._out_spec.children_specs[1].num_leaves
    assert(num_fwd_outputs + num_bwd_outputs == len(output_node.args[0]))
    bwd_outputs = [value_remap[i] for i in output_node.args[0][num_fwd_outputs:]]
    if len(bwd_outputs) == 1:
        bwd_outputs = bwd_outputs[0]
    bw_graph.output(bwd_outputs)
    bw_module = fx.GraphModule(fx_module, bw_graph)

    fw_graph = fx.Graph()
    value_remap = {}
    for node in fx_module.graph.nodes:
        if node not in bw_nodes and node.op != 'output':
            value_remap[node] = fw_graph.node_copy(node, lambda n : value_remap[n])

    fwd_outputs = [value_remap[i] for i in output_node.args[0][:num_fwd_outputs]] + [value_remap[n] for n in saved_nodes]
    if len(fwd_outputs) == 1:
        fwd_outputs = fwd_outputs[0]
    fw_graph.output(fwd_outputs)
    fw_module = fx.GraphModule(fx_module, fw_graph)
    return fw_module, bw_module

import tvm
from tvm import relay
from tvm.contrib import graph_executor


def compiled_function(fn):
    """Wraps a jax function to be supported by PyTorch.
    The resulting PyTorch autograd.Function is only differentiable once.
    We could write a "twice_differentiable_jax_function" to enable second order
    derivatives with autograd.
    """
    compiled_fw = None
    compiled_bw = None
    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            nonlocal compiled_fw, compiled_bw
            if compiled_fw is None:
                out = fn(*args)
                with torch.enable_grad():
                    fx_g = make_fx(vjpfull)(fn, args, (torch.ones_like(out),))
                fw_module, bw_module = partition_backwards(fx_g)
                garbage_hack = torch.randn(())
                fw_args = (garbage_hack,) + args

                # compiled_fw = torch.jit.trace(fw_module, fw_args)
                # shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(fw_args)]
                # mod, params = relay.frontend.from_pytorch(compiled_fw, shape_list)


                # target = tvm.target.Target("llvm", host="llvm")
                # dev = tvm.cpu(0)
                # with tvm.transform.PassContext(opt_level=3):
                #     lib = relay.build(mod, target=target, params=params)
                # dtype = "float32"
                # m = graph_executor.GraphModule(lib["default"](dev))
                # m.run()

                compiled_fw = nnc_compile(fw_module, (garbage_hack,) + args)

                fw_outs = compiled_fw(garbage_hack, *args)
                if not isinstance(fw_outs, list):
                    fw_outs = [fw_outs]
                bw_args = fw_outs[1:] + [torch.ones_like(fw_outs[0])]

                # compiled_bw = torch.jit.trace(bw_module, bw_args)
                compiled_bw = nnc_compile(bw_module, bw_args)
            garbage_hack = torch.randn(())
            fw_outs = compiled_fw(garbage_hack, *args)
            ctx.activations = fw_outs[1:]
            # import pdb; pdb.set_trace()
            return fw_outs[0]

        @staticmethod
        def backward(ctx, *args):
            out = compiled_bw(*ctx.activations, args[0].contiguous())
            return tuple(out)

    return CompiledFunction


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1,1)
    def forward(self, x):
        output = self.fc(x)
        def f(x):
            return (x*2).sin()
        f = compiled_function(f).apply
        return f(output)


def f(a, b):
    return (a * b).sum()
nnc_f = compiled_function(f).apply
a = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
iters = 100
nnc_f(a, b)
def bench(func):
    begin = time.time()
    for _ in range(iters):
        out = func(a, b)
        # out.sum().backward()
    print(time.time()-begin)

bench(f)
bench(nnc_f)
