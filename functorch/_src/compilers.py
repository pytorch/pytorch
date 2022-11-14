import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.fx as fx
import torch.nn as nn
from torch._decomp import get_decompositions
import torch.utils._pytree as pytree

from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)
import torch.utils._pytree as pytree



# These canonicalizations are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
def _canonicalize(fx_g):
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten._to_copy:
            node.target = torch.ops.aten.to
    fx_g.recompile()
    return fx_g


@contextmanager
def _disable_jit_autocast():
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
    try:
        yield
    finally:
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)


@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """

    with _disable_jit_autocast():
        strip_overloads(fx_g)

        for node in fx_g.graph.nodes:
            if (
                node.target == torch.ops.aten._to_copy
                and len(node.args) == 1
                and len(node.kwargs) == 1
                and "dtype" in node.kwargs
            ):
                node.target = torch.ops.aten.to

        for node in fx_g.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs

        fx_g.graph.lint()

        fx_g.recompile()

        f = torch.jit.script(fx_g)

        torch._C._jit_pass_remove_mutation(f.graph)

        f = torch.jit.freeze(f.eval())
        f = torch.jit.optimize_for_inference(f)
        if not any(isinstance(t, torch._subclasses.FakeTensor) for t in inps):
            f(*inps)
    return f


@make_boxed_compiler
def _draw_graph_compile(fx_g, _, name, clear_meta=True):
    print(fx_g.code)
    draw_graph(fx_g, name, clear_meta=clear_meta)
    return fx_g


def draw_graph_compile(name):
    return partial(_draw_graph_compile, name=name)


@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """
    return fx_g

class DebugInterpreter(fx.Interpreter):
    def run_node(self, n):
        # TODO: This will fail once we start caching in AOTAutograd
        # again, because we need to remap SymInts to their new values
        # in the presence of dynamism
        def check(nv, rv, desc):
            assert callable(desc)
            assert nv.size() == rv.size(), f"{desc()}: {nv.size()} != {rv.size()}"
            assert nv.dtype == rv.dtype, f"{desc()}: {nv.dtype} != {rv.dtype}"
            same_strides, idx = torch._prims_common.check_significant_strides(nv, rv, only_cuda=False)
            assert same_strides, f"{desc()}: {nv.stride()} != {rv.stride()} (mismatch at index {idx})"

        # Check that inputs are consistent
        # NB: I don't think this is actually necessary.
        """
        a_vals, _ = pytree.tree_flatten((n.args, n.kwargs))
        for i, a in enumerate(a_vals):
            if not isinstance(a, fx.Node):
                continue
            if 'val' not in a.meta:
                continue
            if not isinstance(a.meta['val'], torch.Tensor):
                continue
            r = self.env[a]
            # TODO: make this more descriptive
            check(a.meta['val'], r, lambda: f"input {i}")
        """

        r = super().run_node(n)
        # Check that outputs are consistent
        if 'val' in n.meta:
            n_vals, n_spec = pytree.tree_flatten(n.meta['val'])
            r_vals, r_spec = pytree.tree_flatten(r)
            assert n_spec == r_spec, f"{n_spec} != {r_spec}"
            assert len(n_vals) == len(r_vals), f"{len(n_vals)} != {len(r_vals)}"
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f"output {i}")
        return r


@make_boxed_compiler
def debug_nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """
    return DebugInterpreter(fx_g).run

@make_boxed_compiler
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
    aten.conj_physical,
    aten.is_same_size,
}

default_decompositions = get_decompositions(default_decompositions)


@make_boxed_compiler
def print_compile(fx_g, _):
    print(fx_g.code)
    return fx_g


def memory_efficient_fusion(
    fn: Union[Callable, nn.Module],
    static_argnums: Optional[Tuple[int]] = None,
    **kwargs,
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


graph_index = 0


def get_inputs(input_data_path):
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
    inputs = []
    with (open(input_data_path, "rb")) as f:
        inputs_meta = pickle.load(f)
        inputs = []
        for meta in inputs_meta:
            if len(meta) == 1:
                type = meta
                input = type(random.rand())
            else:
                type, shape, stride, dtype, device = meta
                if dtype in {
                    torch.int,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                    torch.int,
                    torch.uint8,
                    int,
                    float,
                }:
                    input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input = torch.rand(shape, dtype=dtype, device=device)
            inputs.append(input)
    return inputs


def _save_fx_default(current_name, folder_name, dump_example_input, gm, example_inputs):
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish difference graphs
    """
    from functorch.compile import aot_module_simplified

    def get_input_meta(args):
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):  # joint input
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            if type(arg) == int or type(arg) == float:
                input_meta.append((type(arg),))
            else:
                input_meta.append(
                    (type(arg), arg.shape, arg.stride(), arg.dtype, arg.device)
                )
        return input_meta

    def graph_saver_helper(gm_to_save, args, type_name):
        global graph_index
        if len(gm_to_save.graph.nodes) == 0:
            logging.log(
                logging.WARNING,
                f"No nodes in graph {current_name}_{type_name}_{graph_index}.",
            )
            return

        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())  # remove codegen
        gm.recompile()

        input_meta = get_input_meta(args)

        isExist = os.path.exists(f"{folder_name}/{current_name}")
        if not isExist:
            os.makedirs(f"{folder_name}/{current_name}")
        gm.to_folder(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}"
        )
        pickle.dump(
            input_meta,
            open(
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input",  # noqa: B950
                "wb",
            ),
        )  # noqa: E501
        if dump_example_input:
            torch.save(
                args,
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt",  # noqa: B950
            )  # noqa: E501

    def graph_saver_forward(gm, fw_args):
        graph_saver_helper(gm, fw_args, "forward")
        return gm

    def graph_saver_backward(gm, bw_args):
        graph_saver_helper(gm, bw_args, "backward")
        global graph_index
        graph_index += 1
        return gm

    def graph_saver_joint(gm, joint_args):
        graph_saver_helper(gm, joint_args, "joint")
        return default_partition(gm, joint_args)

    return aot_module_simplified(
        gm,
        fw_compiler=graph_saver_forward,
        bw_compiler=graph_saver_backward,
        partition_fn=graph_saver_joint,
        decompositions=default_decompositions,
    )


def graph_dumper_aot(current_name, folder_name, dump_example_input=False):
    """
    Dump the forward, backward, and joint computation graph.
    Example Usage:
    save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input = False)
    optimize_ctx = torchdynamo.optimize(
        save_fx_func
    )
    with torch.enable_grad():
        with optimize_ctx:
            result = forward_and_backward_pass(model, example_inputs)
    """
    global graph_index
    graph_index = 0
    return partial(_save_fx_default, current_name, folder_name, dump_example_input)
