import torch
from functools import partial
from .aot_autograd import draw_graph
import time


def ts_compile(fx_g, _):
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten.new_zeros:
            if node.args[1] == []:
                args = list(node.args)
                args[1] = [1]
                node.args = tuple(args)
    fx_g.graph.lint()
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
    graph = f.graph
    outputs = list(graph.outputs())
    output = outputs[0]
    graph.eraseOutput(0)
    outputs = list(output.node().inputs())
    for inp in output.node().inputs():
        graph.registerOutput(inp)
    output.node().destroy()
    torch._C._jit_pass_remove_mutation(graph)
    for i in range(len(list(graph.outputs()))):
        graph.eraseOutput(0)
    node = graph.create("prim::ListConstruct", outputs)
    graph.appendNode(node)
    node.output().setType(torch._C.ListType.ofTensors())
    graph.registerOutput(node.output())
    torch._C._jit_pass_remove_mutation(f.graph)

    f = torch.jit.freeze(f.eval())
    f = torch.jit.optimize_for_inference(f)
    return f


def _draw_graph_compile(fx_g, _, name):
    draw_graph(fx_g, name)
    return fx_g


def draw_graph_compile(name):
    return partial(draw_graph_compile, name=name)


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
    print(f.code)
    return f
