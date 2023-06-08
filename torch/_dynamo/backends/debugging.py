import functools
import traceback
from importlib import import_module

from functorch.compile import min_cut_rematerialization_partition

import torch
from torch._functorch.compilers import ts_compile
from .. import config
from .common import aot_autograd
from .registry import register_debug_backend as register_backend

"""
This file contains TorchDynamo backends intended for debugging uses.
"""


@register_backend
def eager(gm, fake_tensor_inputs):
    return gm


@register_backend(name="explain")
def explain(gm: torch.fx.GraphModule, example_inputs):
    """
    Explain `compiler_fn` that provide explanation for graph breaks and logs modules' graph information.
    This function prints the graph structure if `config.explain_print_graphs` is set to True.
    It uses the `tabulate` library to display the node information in a table format.
    If `tabulate` is not installed, it prints an error message suggesting to install it using `pip install tabulate`.

    In case of any graph break while compilation,
    it prints the break reason and the formatted stack trace for debugging-purposes.

    Args:
        gm (torch.fx.GraphModule): The module for debugging.
        example_inputs: Inputs to the module.

    Returns:
        gm.forward as callable.

    Example usage:
        ```
        opt_fn = torch.compile(fn,backend="explain") #  or torch._dynamo.optimize("explain")(fn)
        ```
    """
    if config.explain_print_graphs:
        try:
            from tabulate import tabulate
        except ImportError:
            print(
                "`Explain Backend` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library."
            )

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        print(
            "Node:\n",
            tabulate(
                node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
            ),
        )
    if gm.compile_subgraph_reason.graph_break:
        formatted_stack = "".join(
            traceback.format_list(gm.compile_subgraph_reason.user_stack)
        )
        msg = f"{gm.compile_subgraph_reason.reason}\n{formatted_stack}"
        print("Graph Break\nBreak reason:\n", msg)

    return gm.forward


@register_backend
def eager_debug(gm, fake_tensor_inputs):
    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # We could add more debugging bits here.
    # Right now, this backend can be used to check for and error on
    # custom dispatcher ops that have incorrect schemas.
    def inner(*args):
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner


@register_backend(name="ts")
def torchscript(gm, fake_tensor_inputs):
    return torch.jit.script(gm)


# used boxed call to discard inputs when they are no longer needed
def boxed_nop(fx_g, example_inputs):
    def run(args):
        return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True
    return run


# Useful for debugging purpose
# aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
aot_eager = aot_autograd(fw_compiler=boxed_nop)
register_backend(name="aot_eager", compiler_fn=aot_eager)

# Uses TorchInductor AOT Autograd decomps and partitioner to isolate aot vs
# inductor problems.
# aot_eager_decomp_partition just replaces the inductor compiler with nop to help
# isolate inductor vs aot_eager errors
aot_eager_decomp_partition = aot_autograd(
    # these are taken from memory_efficient_fusion()
    fw_compiler=boxed_nop,
    bw_compiler=boxed_nop,
    # NB: lambda here is to delay import of inductor
    decompositions=lambda: import_module(
        "torch._inductor.compile_fx"
    ).select_decomp_table(),
    partition_fn=functools.partial(
        min_cut_rematerialization_partition, compiler="inductor"
    ),
)
register_backend(
    name="aot_eager_decomp_partition", compiler_fn=aot_eager_decomp_partition
)

# AOT Autograd with torchscript backend. Default partitioner.
# aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
# by using the relevant fuser with torch.jit.fuser(...)
aot_ts = aot_autograd(fw_compiler=ts_compile)
register_backend(name="aot_ts", compiler_fn=aot_ts)


# These buggy backends are used for inducing bugs so that we can test
# our repro extraction / minifier scripts


class ReluCompileError(Exception):
    pass


class TestingOnlyCompileError(Exception):
    pass


@register_backend
def relu_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise ReluCompileError()
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # Require at least one non-trivial thing in the graph,
    # see https://github.com/pytorch/pytorch/issues/102898
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError()
    return gm
