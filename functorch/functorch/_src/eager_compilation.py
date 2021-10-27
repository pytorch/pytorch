from functorch import make_fx
import time
import torch
import torch.nn as nn
from functorch import make_functional_with_buffers, make_fx
from torch.fx.node import map_arg
import torch.fx as fx
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch.fx.passes import graph_drawer
import os

def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = "fx_graph"):
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
    g = graph_drawer.FxGraphDrawer(traced, figname)
    x = g.get_main_dot_graph()
    getattr(x, "write_" + ext.lstrip("."))(fname)

# todo(chilli): clean this up/make it more understandable
def default_partition(fx_module: fx.GraphModule, _joint_inputs):
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

def partition_with_recompute_fwd_in_bwd(joint_module: fx.GraphModule, _joint_inputs):
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recopmuting helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimintation.
    """

    def _extract_graph_with_given_outputs(joint_graph, outputs, is_fwd=False):
        """
        Returns a copy of joint_graph with given outputs.

        If its forward graph, we need extra bookkeeping
            1) Remove tangent nodes in the input.
            2) Pass the inputs directly to the output. This will be saved in the
            backward ctx.
        """
        # Set up val_map to be used later for copying the graph
        val_map = {}
        saved_nodes = []
        if is_fwd:
            # Remove the tangent placeholder nodes from the graph
            def _tangent_finder(node):
                return node.op == "placeholder" and "tangents" in node.target
            tangent_nodes = filter(_tangent_finder, joint_graph.nodes)
            for tangent_node in tangent_nodes:
                val_map[tangent_node] = 1

            # Find the saved tensor nodes that will be used by ctx later
            def _placeholder_finder(node):
                return node.op == "placeholder" and "tangents" not in node.target
            saved_nodes = list(filter(_placeholder_finder, joint_graph.nodes))

        # Make a copy of the joint graph
        graph = fx.Graph()
        graph.graph_copy(joint_graph, val_map)

        # Set the outputs
        outputs = outputs + saved_nodes
        if len(outputs) == 1:
            graph.output(val_map[outputs[0]])
        else:
            graph.output([val_map[out] for out in outputs])

        # Run dead code elimination to remove unnecessary nodes
        graph.eliminate_dead_code()
        graph.lint()
        return graph

    # Find the output node
    output_node = None
    for n in reversed(joint_module.graph.nodes):
        if n.op == "output":
            output_node = n
            break

    # Get the forward and backward output nodes
    num_fwd_outputs = joint_module._out_spec.children_specs[0].num_leaves
    fwd_outputs = output_node.args[0][0:num_fwd_outputs]
    bwd_outputs = output_node.args[0][num_fwd_outputs:]

    # Construct the forward module
    fwd_graph = _extract_graph_with_given_outputs(
        joint_module.graph, fwd_outputs, is_fwd=True
    )
    fwd_module = fx.GraphModule(joint_module, fwd_graph)

    # Construct the backward module
    bwd_graph = _extract_graph_with_given_outputs(joint_module.graph, bwd_outputs)
    bwd_module = fx.GraphModule(joint_module, bwd_graph)

    return fwd_module, bwd_module

def create_joint_forward_backward(fn):
    def joint_forward_backward(primals, tangents):
        out = fn(*primals)
        primals = [p for p in pytree.tree_flatten(primals)[0] if p.requires_grad]
        backward_out = []
        if primals:
            backward_out = torch.autograd.grad(out, primals, grad_outputs=tangents, create_graph=True, allow_unused=True)
        return out, backward_out
    return joint_forward_backward

def draw_joint_graph(graph, joint_inputs, file_name="full_graph.png"):
    draw_graph(graph, file_name)
    return default_partition(graph, joint_inputs)

def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]

def create_compiled_function(flat_fn, fw_compiler, bw_compiler, partition_fn):
    joint_forward_backward = create_joint_forward_backward(flat_fn)

    compiled_fw = None
    compiled_bw = None
    num_outs = None

    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *flat_args):
            nonlocal compiled_fw, compiled_bw, num_outs
            if compiled_fw is None:
                out = flat_fn(*flat_args)
                if isinstance(out, (list, tuple)):
                    num_outs = len(out)
                else:
                    num_outs = 1

                joint_inputs = (flat_args, out)
                with torch.enable_grad():
                    fx_g = make_fx(joint_forward_backward)(*joint_inputs)
                fw_module, bw_module = partition_fn(fx_g, joint_inputs)
                # print(fw_module.code, bw_module.code)

                compiled_fw = fw_compiler(fw_module, flat_args)
                fw_outs = normalize_as_list(compiled_fw(*flat_args))

                bw_args = fw_outs[num_outs:] + fw_outs[0:num_outs]
                compiled_bw = bw_compiler(bw_module, bw_args)
            fw_outs = normalize_as_list(compiled_fw(*flat_args))
            ctx.save_for_backward(*fw_outs[num_outs:])
            if num_outs == 1:
                return fw_outs[0]
            return tuple(fw_outs[0:num_outs])

        @staticmethod
        def backward(ctx, *flat_args):
            # hmm... this doesn't feel right. todo
            contiguous_args = [t.contiguous() for t in flat_args]
            out = normalize_as_list(compiled_bw(*ctx.saved_tensors, *contiguous_args))
            out_iter = iter(out)
            grad_out = [next(out_iter) if p else None for p in ctx.needs_input_grad]
            return tuple(grad_out)

    return CompiledFunction


# using this reduces the overhead by about 50%
# import tree
def compiled_function(fn, fw_compiler, bw_compiler, partition_fn=default_partition):
    saved_fn = None

    def returned_function(*args, **kwargs):
        nonlocal saved_fn
        # flattened_args = tree.flatten((args, kwargs))
        flattened_args, _ = pytree.tree_flatten((args, kwargs))

        if saved_fn is None:
            flattened_args, args_spec = pytree.tree_flatten((args, kwargs))
            def flat_fn(*args):
                args, kwargs = pytree.tree_unflatten(args, args_spec)
                return fn(*args, **kwargs)

            saved_fn = create_compiled_function(flat_fn, fw_compiler, bw_compiler, partition_fn).apply
        return saved_fn(*flattened_args)

    return returned_function


def tvm_compile(fx_module, example_inputs, name = None):
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor
    import os

    jit_mod = torch.jit.script(fx_module)
    # jit_mod = torch.jit.trace(fx_module, example_inputs)

    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params, target)
    for task in tasks:
        print(task.compute_dag)
    if name is None:
        log_file = f'{time.time()}.json'
    else:
        log_file = f'{name}.json'
    if len(tasks) != 0:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        if not os.path.exists(log_file):
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=10000,  # change this to 20000 to achieve the best performance
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                # early_stopping=1000,
                # verbose=2,
            )
            tuner.tune(tune_option)

    dev = tvm.cpu(0)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:

                m.set_input(f"inp_{idx}", tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg)))
        m.run()
        outs = [torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack()) for i in range(m.get_num_outputs())]
        return outs
    return exec_tvm

def tvm_function(fn, name):
    return compiled_function(fn, partial(tvm_compile, name=f'fw_{name}'), partial(tvm_compile, name=f'bw_{name}'))

def compiled_module(mod, fw_compiler, bw_compiler, partition_fn=default_partition):
    func_mod, params, buffers = make_functional_with_buffers(mod)
    compiled_f = compiled_function(func_mod, fw_compiler, bw_compiler, partition_fn)

    class CompiledModule(nn.Module):
        def __init__(self):
            super(CompiledModule, self).__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                tuple(self.orig_module.parameters()),
                tuple(self.orig_module.buffers()),
                *args,
                **kwargs
            )

    return CompiledModule()
