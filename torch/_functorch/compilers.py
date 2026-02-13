from __future__ import annotations

import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import sympy

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import SymInt
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols

from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch.fx.node import Node
    from torch.types import IntLikeType


_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)


# These canonicalization are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
def _canonicalize(fx_g: fx.GraphModule) -> fx.GraphModule:
    for node in fx_g.graph.find_nodes(
        op="call_function", target=torch.ops.aten._to_copy
    ):
        node.target = torch.ops.aten.to
    fx_g.recompile()
    return fx_g


@contextmanager
def _disable_jit_autocast() -> Generator[None, None, None]:
    # pyrefly: ignore [missing-attribute]
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
    try:
        yield
    finally:
        # pyrefly: ignore [missing-attribute]
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)


@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps: Sequence[Any]) -> torch.jit.ScriptModule:
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

        for node in fx_g.graph.find_nodes(
            op="call_function", target=torch.ops.aten._to_copy
        ):
            if len(node.args) == 1 and len(node.kwargs) == 1 and "dtype" in node.kwargs:
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

        # pyrefly: ignore [missing-attribute]
        torch._C._jit_pass_remove_mutation(f.graph)

        f = torch.jit.freeze(f.eval())
        f = torch.jit.optimize_for_inference(f)
        if not any(isinstance(t, torch._subclasses.FakeTensor) for t in inps):
            f(*inps)
    return f


def _draw_graph_compile(
    fx_g: fx.GraphModule, _: Any, name: str, clear_meta: bool = True
) -> fx.GraphModule:
    print(fx_g.code)
    draw_graph(fx_g, name, clear_meta=clear_meta)
    return fx_g


def draw_graph_compile(
    name: str,
) -> Callable[[fx.GraphModule, list[Any]], fx.GraphModule]:
    return make_boxed_compiler(partial(_draw_graph_compile, name=name))


@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """
    return fx_g


class DebugInterpreter(fx.Interpreter):
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self.symbol_mapping = bind_symbols(
            # pyrefly: ignore[bad-argument-type]
            self.module,
            *args,
        )
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: Node) -> Any:
        def subst_symint(ni: IntLikeType) -> int:
            if not isinstance(ni, SymInt):
                return ni
            r = sympy.expand(ni.node.expr.xreplace(self.symbol_mapping))
            if not r.is_number:
                raise AssertionError(f"expected r to be a number, got {r}")
            return int(r)

        def subst_symint_tuple(nis: tuple[IntLikeType, ...]) -> tuple[int, ...]:
            return tuple(subst_symint(ni) for ni in nis)

        def check_significant_strides(a: torch.Tensor, b: torch.Tensor) -> bool:
            if subst_symint(a.numel()) > 0:
                for idx in range(a.ndim):
                    if (
                        subst_symint(a.stride(idx)) != b.stride(idx)
                        and subst_symint(a.size(idx)) > 1
                    ):
                        return False
            return True

        def check(nv: torch.Tensor, rv: torch.Tensor, desc: Callable[[], str]) -> None:
            if not callable(desc):
                raise AssertionError(f"expected desc to be callable, got {type(desc)}")
            if nv.dtype != rv.dtype:
                raise AssertionError(f"{desc()}: {nv.dtype} != {rv.dtype}")
            if subst_symint_tuple(nv.size()) != rv.size():
                raise AssertionError(
                    f"{desc()}: {nv.size()} aka {subst_symint_tuple(nv.size())} != {rv.size()}"
                )
            same_strides = check_significant_strides(nv, rv)
            if not same_strides:
                raise AssertionError(
                    f"{desc()}: {nv.stride()} aka {subst_symint_tuple(nv.stride())} != {rv.stride()}"
                )

        r = super().run_node(n)
        if "val" in n.meta:
            n_vals, _n_spec = pytree.tree_flatten(n.meta["val"])
            r_vals, _r_spec = pytree.tree_flatten(r)
            # TODO: There is some sort of problem where we record that an
            # operator returned a tuple/list, and then later it turns out the
            # real version of the operator returned a list/tuple. Need to
            # figure out what's actually going on here, the error itself is
            # harmless enough as we only getitem out the outputs.
            # assert n_spec == r_spec, f"{n_spec} != {r_spec}"
            if len(n_vals) != len(r_vals):
                raise AssertionError(f"{len(n_vals)} != {len(r_vals)}")
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f"output {i} where {self.symbol_mapping}")
        return r


@make_boxed_compiler
def debug_nop(
    fx_g: fx.GraphModule, _: Any
) -> Callable[[DebugInterpreter, Any, dict[Node, Any] | None, bool], Any]:
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """
    return DebugInterpreter(fx_g).run


@make_boxed_compiler
def simple_ts_compile(fx_g: fx.GraphModule, _: Any) -> torch.jit.ScriptModule:
    strip_overloads(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f: Callable[..., Any]) -> Callable[..., Any]:
    return aot_function(f, simple_ts_compile)


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

# pyrefly: ignore[bad-argument-type]
default_decompositions = get_decompositions(default_decompositions)


@make_boxed_compiler
def print_compile(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    print(fx_g.code)
    return fx_g


def memory_efficient_fusion(
    fn: Callable[_P, _R] | nn.Module,
    **kwargs: Any,
) -> Callable[_P, _R] | nn.Module:
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
            that takes one or more arguments. Must return one or more Tensors.
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
    }
    config.update(kwargs)
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)  # pyrefly: ignore[bad-argument-type]
    else:
        return aot_function(fn, **config)  # pyrefly: ignore[bad-argument-type]


def debug_compile(
    fx_g: fx.GraphModule, inps: Sequence[torch.Tensor]
) -> torch.jit.ScriptModule:
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
    from foo import FxModule  # pyrefly: ignore[missing-import]

    FxModule().cuda()(*inps)

    return ts_compile(fx_g, inps)


graph_index: int = 0


def get_inputs(input_data_path: str) -> list[torch.Tensor]:
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
    inputs: list[torch.Tensor] = []
    with open(input_data_path, "rb") as f:
        inputs_meta = pickle.load(f)
        inputs = []
        for meta in inputs_meta:
            if len(meta) == 1:
                type = meta
                input_ = type(random.random())
            else:
                type, shape, _stride, dtype, device = meta
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
                    input_ = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input_ = torch.rand(shape, dtype=dtype, device=device)
            inputs.append(input_)
    return inputs


def _save_fx_default(
    current_name: str,
    folder_name: str,
    dump_example_input: bool,
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> nn.Module:
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

    def get_input_meta(args: Any) -> list[Any]:
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):  # joint input
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            if type(arg) is int or type(arg) is float:
                input_meta.append((type(arg),))
            else:
                input_meta.append(
                    (type(arg), arg.shape, arg.stride(), arg.dtype, arg.device)
                )
        return input_meta

    def graph_saver_helper(
        gm_to_save: fx.GraphModule, args: Any, type_name: str
    ) -> None:
        global graph_index
        if len(gm_to_save.graph.nodes) == 0:
            log.log(
                logging.WARNING,
                "No nodes in graph {%s}_{%s}_{%s}.",
                current_name,
                type_name,
                graph_index,
            )
            return

        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())  # remove codegen
        gm.recompile()

        input_meta = get_input_meta(args)

        os.makedirs(f"{folder_name}/{current_name}", exist_ok=True)
        gm.to_folder(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}"
        )
        with open(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input",
            "wb",
        ) as f:
            pickle.dump(input_meta, f)
        if dump_example_input:
            torch.save(
                args,
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt",  # noqa: B950
            )  # noqa: E501

    def graph_saver_forward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        graph_saver_helper(gm, example_inputs, "forward")
        return gm

    def graph_saver_backward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        graph_saver_helper(gm, example_inputs, "backward")
        global graph_index
        graph_index += 1
        return gm

    def graph_saver_joint(
        gm: fx.GraphModule, joint_args: list[torch.Tensor]
    ) -> tuple[fx.GraphModule, fx.GraphModule]:
        graph_saver_helper(gm, joint_args, "joint")
        return default_partition(gm, joint_args)  # pyrefly: ignore[missing-argument]

    # pyrefly: ignore[bad-return]
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=graph_saver_forward,  # pyrefly: ignore[bad-argument-type]
        bw_compiler=graph_saver_backward,  # pyrefly: ignore[bad-argument-type]
        partition_fn=graph_saver_joint,
        decompositions=default_decompositions,  # pyrefly: ignore[bad-argument-type]
    )


# WARNING: This isn't tested anywhere!!
def graph_dumper_aot(
    current_name: str, folder_name: str, dump_example_input: bool = False
) -> Callable[[bool, nn.Module], Any]:
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
