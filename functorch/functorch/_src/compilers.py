import torch
from functools import partial
from typing import Iterable
from .aot_autograd import draw_graph, aot_function, partition_with_recompute_fwd_in_bwd, aot_module
from .decompositions import decomposition_table
import time


def ts_compile(fx_g, _):
    # print(fx_g.code)
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten.new_zeros:
            if node.args[1] == []:
                args = list(node.args)
                args[1] = [1]
                node.args = tuple(args)

    for node in fx_g.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs

    fx_g.graph.lint()

    # print(set([i.target for i in fx_g.graph.nodes if i.op == 'call_function']))
    # Works around this NVFuser issue: https://github.com/csarofeen/pytorch/issues/1311
    for i in range(1000):
        attr = f'_tensor_constant{i}'
        if hasattr(fx_g, attr):
            setattr(fx_g, attr, getattr(fx_g, attr).cuda())
        else:
            break

    fx_g.recompile()

    f = torch.jit.script(fx_g)

    # Works around alias analysis issues in TS
    # graph = f.graph
    # outputs = list(graph.outputs())
    # output = outputs[0]
    # graph.eraseOutput(0)
    # outputs = list(output.node().inputs())
    # for inp in output.node().inputs():
    #     graph.registerOutput(inp)
    # output.node().destroy()
    # torch._C._jit_pass_remove_mutation(graph)
    # for i in range(len(list(graph.outputs()))):
    #     graph.eraseOutput(0)
    # node = graph.create("prim::ListConstruct", outputs)
    # graph.appendNode(node)
    # node.output().setType(torch._C.ListType.ofTensors())
    # graph.registerOutput(node.output())
    torch._C._jit_pass_remove_mutation(f.graph)

    f = torch.jit.freeze(f.eval())
    f = torch.jit.optimize_for_inference(f)
    return f


def tensorexpr_compile(fx_module, flat_args):
    """Compiles the given fx_module using TensorExpr Kernel"""
    inp_devices = set([i.device for i in flat_args if isinstance(i, torch.Tensor)])
    assert len(inp_devices) == 1
    inp_device = list(inp_devices)[0]
    inputs = list()
    output_refs = list()
    for node in fx_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node)
        elif node.op == "output":
            outputs = node.args[0]
            if not isinstance(outputs, Iterable):
                outputs = (outputs,)
            new_outputs = []
            for idx, output in enumerate(outputs):
                # Appends (bool, idx) pairs
                # if True, read from kernel outputs
                # if False, read from kernel inputs
                if output in inputs:
                    output_refs.append((False, inputs.index(output)))
                elif output in outputs[:idx]:
                    output_refs.append((True, output_refs[outputs.index(output)][1]))
                else:
                    output_refs.append((True, len(new_outputs)))
                    new_outputs.append(output)
            node.args = (tuple(new_outputs),)
    fx_module.graph.lint()
    fx_module.recompile()

    for i in range(0, 100):
        attr = f"_tensor_constant{i}"
        if hasattr(fx_module, attr):
            setattr(fx_module, attr, getattr(fx_module, attr).to(inp_device))
        else:
            break

    jit_module = torch.jit.trace(fx_module, flat_args)
    jit_module = torch.jit.freeze(jit_module.eval())
    torch._C._jit_trace_module(jit_module._c, tuple(flat_args))
    torch._C._te.remove_unused_self_argument(jit_module.graph)
    torch._C._te.annotate_input_shapes(jit_module.graph, tuple(flat_args))
    torch._C._jit_pass_lower_all_tuples(jit_module.graph)
    te_kernel = torch._C._te.TensorExprKernel(jit_module.graph)

    def f(*args):
        outs = te_kernel.run(args)
        if not isinstance(outs, tuple) and not isinstance(outs, list):
            outs = (outs,)
        real_outs = []
        for out in output_refs:
            if out[0]:
                real_outs.append(outs[out[1]])
            else:
                real_outs.append(args[out[1]])
        return real_outs

    return f


def _draw_graph_compile(fx_g, _, name):
    draw_graph(fx_g, name)
    return fx_g


def draw_graph_compile(name):
    return partial(_draw_graph_compile, name=name)


def _tvm_compile(fx_module, example_inputs, name=None):
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
    m = graph_executor.GraphModule(lib["default"](dev))

    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:

                m.set_input(f"inp_{idx}", tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg)))
        m.run()
        outs = [torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack()) for i in range(m.get_num_outputs())]
        return outs
    return exec_tvm


def tvm_compile(name):
    return partial(tvm_compile, name=name)


def nop(f, _):
    return f


def simple_ts_compile(fx_g, _):
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f):
    return aot_function(f, simple_ts_compile)


aten = torch.ops.aten
default_decompositions = set([
    aten.detach,
    aten.gelu_backward,
    aten._log_softmax_backward_data,
    aten.leaky_relu_backward,
    aten.sigmoid_backward,
    aten.threshold_backward,
    aten.hardtanh_backward,
    aten.hardsigmoid_backward,
    aten.hardswish_backward,
    aten.tanh_backward,
    aten.silu_backward,
])
default_decompositions = {k: v for k, v in decomposition_table.items() if k in default_decompositions}


def memory_efficient_fusion(fn, static_argnums=None):
    """
    Recomputes the fwd pass in the bwd pass to perform memory efficient fusion.
    Uses NVFuser as the backend compiler.
    """
    config = {
        'fw_compiler': ts_compile,
        'bw_compiler': ts_compile,
        'partition_fn': partition_with_recompute_fwd_in_bwd,
        'hasher_type': "StaticShapheHasher",
        'decompositions': default_decompositions,
        'static_argnums': static_argnums
    }
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)
    else:
        return aot_function(fn, **config)
