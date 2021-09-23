import torch
from .python_trace import make_fx
from torch.fx.node import map_arg
import torch.fx as fx
import torch.utils._pytree as pytree

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

    num_fwd_outputs = fx_module._out_spec.children_specs[0].num_leaves
    num_bwd_outputs = fx_module._out_spec.children_specs[1].num_leaves
    bw_outputs = output_node.args[0][num_fwd_outputs:]

    bw_graph = fx.Graph()
    value_remap = {}
    for saved_node in saved_nodes:
        value_remap[saved_node] = bw_graph.placeholder(saved_node.name)

    for node in fx_module.graph.nodes:
        if node in bw_nodes or node in bw_outputs:
            value_remap[node] = bw_graph.node_copy(node, lambda n : value_remap[n])

    assert(num_fwd_outputs + num_bwd_outputs == len(output_node.args[0]))
    bwd_outputs = [value_remap[i] for i in bw_outputs]
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
    fw_module.graph.lint()
    bw_module.graph.lint()
    return fw_module, bw_module

def create_joint_forward_backward(fn):
    def joint_forward_backward(primals, tangents):
        primals = pytree.tree_map(lambda x: x.requires_grad_(), primals)
        out = fn(*primals)
        backward_out = torch.autograd.grad(out, primals, grad_outputs=tangents, create_graph=True, allow_unused=True)
        return out, backward_out
    return joint_forward_backward

def compiled_function(fn, fw_compiler, bw_compiler):
    fw_module = None
    compiled_fw = None
    bw_module = None
    compiled_bw = None
    num_outs = None

    saved_fn = None

    def returned_function(*args, **kwargs):
        nonlocal saved_fn
        flattened_args, args_spec = pytree.tree_flatten((args, kwargs))

        if saved_fn is None:
            def flat_fn(*args):
                args, kwargs = pytree.tree_unflatten(args, args_spec)
                return fn(*args, **kwargs)

            joint_forward_backward = create_joint_forward_backward(flat_fn)

            class CompiledFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, *args):
                    nonlocal compiled_fw, compiled_bw, fw_module, bw_module, num_outs
                    if compiled_fw is None:
                        out = flat_fn(*args)
                        if isinstance(out, (list, tuple)):
                            num_outs = len(out)
                        else:
                            num_outs = 1
                        with torch.enable_grad():
                            fx_g = make_fx(joint_forward_backward)(args, (out,))
                        fw_module, bw_module = partition_backwards(fx_g)

                        compiled_fw = fw_compiler(fw_module, args)
                        fw_outs = compiled_fw(*fw_module.graph.flatten_inps(args))

                        if not isinstance(fw_outs, list):
                            fw_outs = [fw_outs]

                        bw_args = fw_outs[num_outs:] + fw_outs[0:num_outs]
                        compiled_bw = bw_compiler(bw_module, bw_args)

                    fw_outs = compiled_fw(*fw_module.graph.flatten_inps(args))
                    if not isinstance(fw_outs, list):
                        fw_outs = [fw_outs]
                    ctx.activations = fw_outs[num_outs:]
                    if num_outs == 1:
                        return fw_outs[0]
                    return tuple(fw_outs[0:num_outs])

                @staticmethod
                def backward(ctx, *args):
                    contiguous_args = [t.contiguous() for t in args]
                    out = compiled_bw(*ctx.activations, *contiguous_args)
                    if not isinstance(out, list):
                        out = [out]
                    return tuple(out)
            saved_fn = CompiledFunction.apply
        return saved_fn(*flattened_args)

    return returned_function
