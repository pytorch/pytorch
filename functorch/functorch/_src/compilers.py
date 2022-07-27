import torch
import torch.fx as fx
import torch.nn as nn
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

from .aot_autograd import aot_function, aot_module
from .decompositions import get_decompositions
from .partitioners import draw_graph, min_cut_rematerialization_partition
from .compile_utils import strip_overloads
import time


# These canonicalizations are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
def _canonicalize(fx_g):
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten._to_copy:
            node.target = torch.ops.aten.to
    fx_g.recompile()
    return fx_g


def ts_compile(fx_g: fx.GraphModule, _) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """
    for node in fx_g.graph.nodes:
        if (node.target == torch.ops.aten._to_copy and len(node.args) == 1
           and len(node.kwargs) == 1 and 'dtype' in node.kwargs):
            node.target = torch.ops.aten.to

    for node in fx_g.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs

    strip_overloads(fx_g)

    fx_g.graph.lint()

    fx_g.recompile()

    f = torch.jit.script(fx_g)

    torch._C._jit_pass_remove_mutation(f.graph)

    f = torch.jit.freeze(f.eval())
    f = torch.jit.optimize_for_inference(f)
    return f


def tensorexpr_compile(fx_module: fx.GraphModule, flat_args) -> Callable:
    """Compiles the given fx_module using TensorExpr Kernel"""
    inp_devices = {i.device for i in flat_args if isinstance(i, torch.Tensor)}
    assert len(inp_devices) == 1
    inp_device = list(inp_devices)[0]
    inputs = []
    output_refs = []
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


def _draw_graph_compile(fx_g, _, name, clear_meta=True):
    print(fx_g.code)
    draw_graph(fx_g, name, clear_meta=clear_meta)
    return fx_g


def draw_graph_compile(name):
    return partial(_draw_graph_compile, name=name)


def _tvm_compile(
    fx_module, example_inputs, target=None, tuning_logfile=None, use_ansor_tuning=False
):
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor
    import os

    # Find the target and device for TVM.
    dev = tvm.cpu(0)
    if target is None:
        raise ValueError("Setup the TVM target correctly.")
    elif isinstance(target, str):
        if "cuda" in target:
            dev = tvm.cuda(0)
        target = tvm.target.Target(target)
    elif isinstance(target, tvm.target.target.Target):
        if "cuda" in target.keys:
            dev = tvm.cuda(0)

    # JIT the model and pass it to Torchscript to Relay frontend parser. TVM
    # tutorials suggest tracing instead of scripting. The main reason is to
    # avoid Pythonic computation to show up in JIT module. However, with Python
    # key tracing, AOT Autograd leads to simpler graphs. Therefore, we use
    # scripting here to retrieve the JIT module.
    jit_mod = torch.jit.script(fx_module)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)

    # TVM Autotuning
    if use_ansor_tuning:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        if tuning_logfile is None:
            log_file = f"{time.time()}.json"
        else:
            log_file = f"{tuning_logfile}.json"
        if len(tasks) != 0:
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20000,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                # early_stopping=1000,
                # verbose=2,
            )
            tuner.tune(tune_option)
    elif tuning_logfile is not None:
        log_file = f"{tuning_logfile}.json"

    if use_ansor_tuning or tuning_logfile is not None:
        assert os.path.exists(log_file)
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    # Get a graph executor graph module
    m = graph_executor.GraphModule(lib["default"](dev))

    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
                )
        m.run()
        outs = [
            torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
            for i in range(m.get_num_outputs())
        ]
        return outs

    return exec_tvm


def tvm_compile(target, tuning_logfile=None, use_ansor_tuning=False):
    return partial(
        _tvm_compile,
        target=target,
        tuning_logfile=tuning_logfile,
        use_ansor_tuning=use_ansor_tuning,
    )


def nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """
    return fx_g


def simple_ts_compile(fx_g, _):
    strip_overloads(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f, static_argnums=None):
    return aot_function(f, simple_ts_compile, static_argnums=static_argnums)


aten = torch.ops.aten
default_decompositions = {
    aten.detach,
    aten.gelu_backward,
    aten.leaky_relu_backward,
    aten.sigmoid_backward,
    aten.threshold_backward,
    aten.hardtanh_backward,
    aten.hardsigmoid_backward,
    aten.hardswish_backward,
    aten.tanh_backward,
    aten.silu_backward,
    aten.elu_backward,
    aten.cudnn_batch_norm,
    aten.cudnn_batch_norm_backward,
    aten.masked_fill.Scalar,
    aten.masked_fill.Tensor,
    aten.elu,
    aten.leaky_relu,
    aten.hardtanh,
    aten.hardswish,
    aten.hardsigmoid,
}

default_decompositions = get_decompositions(default_decompositions)


def print_compile(fx_g, _):
    print(fx_g.code)
    return fx_g


def memory_efficient_fusion(
    fn: Union[Callable, nn.Module], static_argnums: Optional[Tuple[int]] = None, **kwargs
):
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one ore more arguments. Must return one or more Tensors.
        static_argnums (Optional[Tuple[Int]]): An option tuple of ints to mark
            the arguments of the function as static.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.

    """
    config = {
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
        "hasher_type": "StaticShapeHasher",
        "decompositions": default_decompositions,
        "static_argnums": static_argnums,
    }
    config.update(kwargs)
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)
    else:
        return aot_function(fn, **config)


def debug_compile(fx_g, inps):
    fx_g.to_folder("foo")
    print(
        f"""
##############################################################
# To minimize FX graph, copy and paste the below and run it  #
##############################################################

import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess

inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]
from foo import FxModule
mod = FxModule().cuda()

with torch.jit.fuser("fuser2"):
  # check_nvfuser_subprocess can be replaced with check_nvfuser_correctness_subprocess
  minifier(fx.symbolic_trace(mod), inps, check_nvfuser_subprocess)
"""
    )
    from foo import FxModule

    FxModule().cuda()(*inps)

    return ts_compile(fx_g, inps)
