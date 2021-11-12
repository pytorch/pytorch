from functorch._src.python_key import pythonkey_decompose
import torch
from torch import fx
from torch.fx.proxy import GraphAppendingTracer
from typing import Iterable
from .eager_compilation import compiled_function, partition_with_recompute_fwd_in_bwd

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


def torchscript_nnc_compile(fx_module, flat_args):
    """Compiles the given fx_module using torchscript"""
    traced_module = torch.jit.trace(fx_module, flat_args)
    frozen_module = torch.jit.freeze(traced_module.eval())
    return frozen_module


def torchscript_nvfuser_compile(fx_module, flat_args):
    """Compiles the given fx_module using torchscript nvfuser"""
    if not torch._C._jit_nvfuser_enabled():
        raise RuntimeError("Wrap the call with `with jit.fuser(\"fuser2\") to turn nvfuser on")
    scripted_module = torch.jit.script(fx_module)
    frozen_module = torch.jit.freeze(scripted_module.eval())
    return frozen_module


def torchscript_nnc_operator_authoring(fn, partition_fn):
    fw_compiler = torchscript_nnc_compile
    bw_compiler = torchscript_nnc_compile
    return compiled_function(fn, fw_compiler, bw_compiler, partition_fn, decompose=True)


def torchscript_nvfuser_operator_authoring(fn, partition_fn):
    fw_compiler = torchscript_nvfuser_compile
    bw_compiler = torchscript_nvfuser_compile
    return compiled_function(fn, fw_compiler, bw_compiler, partition_fn, decompose=True)


def tensorexpr_operator_authoring(fn, partition_fn):
    fw_compiler = tensorexpr_compile
    bw_compiler = tensorexpr_compile
    return compiled_function(fn, fw_compiler, bw_compiler, partition_fn, decompose=True)


def memory_efficient_operator_authoring(fn, compiler_name="torchscript_nnc"):
    if compiler_name == "torchscript_nnc":
        return torchscript_nnc_operator_authoring(
            fn, partition_with_recompute_fwd_in_bwd
        )
    elif compiler_name == "tensorexpr_nnc":
        return tensorexpr_operator_authoring(fn, partition_with_recompute_fwd_in_bwd)
    elif compiler_name == "torchscript_nvfuser":
        return torchscript_nvfuser_operator_authoring(
            fn, partition_with_recompute_fwd_in_bwd
        )
    return NotImplementedError(f"{compiler_name} is not implemented")
