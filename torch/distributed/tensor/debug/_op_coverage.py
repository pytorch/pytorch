# mypy: allow-untyped-defs
from operator import itemgetter
from typing import List

import torch
import torch.fx
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed.tensor import DTensor


inductor_decomps = select_decomp_table()

graphs: List[torch.fx.GraphModule] = []


def fwd_bwd_compiler(fx_g, _):
    graphs.append(fx_g)
    return make_boxed_func(fx_g)


def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    output = compiled_mod(*args, **kwargs)

    if output.ndim != 0:
        # if output is not a scalar tensor, by default sum it in order to
        # run backward
        output = output.sum()

    output.backward()

    # one fwd, one bwd graph
    assert len(graphs) == 2
    return graphs


def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    """
    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    """
    # python module required for summary
    import csv

    from tabulate import tabulate

    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)

    op_counts = {}

    for node in fwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    for node in bwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    op_infos = []

    for op, count in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])

    # sort the op info base on the total count index
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)

    headers = ["Operator", "Schema", "Total Count", "Supported"]
    print(tabulate(op_infos, headers=headers))

    if output_csv:
        # Open a CSV file for writing
        with open("op_summary.csv", "w", newline="") as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(headers)
            # Write each table row to the CSV file
            for row in op_infos:
                csv_writer.writerow(row)
