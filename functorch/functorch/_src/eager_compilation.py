import time
import torch
import torch.nn as nn
from functorch import make_fx, grad, nnc_jit, nnc_compile, vmap, make_nnc, vjp
from torch.fx.node import map_arg
import torch.fx as fx
from functools import partial
import os
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
    return fw_module, bw_module


def tvm_compile(fx_module, example_inputs, name = None):
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor
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
                num_measure_trials=100,  # change this to 20000 to achieve the best performance
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
        begin = time.time()
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(f"inp_{idx}", arg)
        m.run()
        outs = [torch.from_numpy(m.get_output(i).numpy()) for i in range(m.get_num_outputs())]
        return outs
    return exec_tvm

def compiled_function(fn, fw_compiler, bw_compiler):
    fw_module = None
    compiled_fw = None
    bw_module = None
    compiled_bw = None

    def vjpfull(primals, tangents):
        out, vjpfn = vjp(fn, *primals)
        return out, vjpfn(*tangents)

    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            nonlocal compiled_fw, compiled_bw, fw_module, bw_module
            if compiled_fw is None:
                out = fn(*args)
                with torch.enable_grad():
                    fx_g = make_fx(vjpfull)(args, (torch.randn_like(out),))
                fw_module, bw_module = partition_backwards(fx_g)


                compiled_fw = fw_compiler(fw_module, args)
                fw_outs = compiled_fw(*fw_module.graph.flatten_inps(args))

                if not isinstance(fw_outs, list):
                    fw_outs = [fw_outs]

                bw_args = fw_outs[1:] + [torch.ones_like(fw_outs[0])]
                compiled_bw = bw_compiler(bw_module, bw_args)

            fw_outs = compiled_fw(*fw_module.graph.flatten_inps(args))
            if not isinstance(fw_outs, list):
                fw_outs = [fw_outs]
            ctx.activations = fw_outs[1:]
            return fw_outs[0]

        @staticmethod
        def backward(ctx, *args):
            out = compiled_bw(*ctx.activations, args[0].contiguous())
            if not isinstance(out, list):
                out = [out]
            return tuple(out)

    return CompiledFunction

def tvm_function(fn, name):
    return compiled_function(fn, partial(tvm_compile, name=f'fw_{name}'), partial(tvm_compile, name=f'bw_{name}'))
