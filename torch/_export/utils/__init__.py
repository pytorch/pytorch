import copy
import dataclasses
import weakref
import re
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Optional, Dict, Union

import sympy

import torch
import torch._dynamo
import torch.fx
import torch.utils._pytree as pytree
import torch._export

from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

def _unlift(gm, inp_pos_to_buffer_name, in_spec, out_spec, state_dict):
    count = 0
    # Step 1: make lifted params as get_attr
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if count in inp_pos_to_buffer_name:
                with gm.graph.inserting_after(node):
                    getattr_node = gm.graph.get_attr(inp_pos_to_buffer_name[count])
                    node.replace_all_uses_with(getattr_node)
                    gm.graph.erase_node(node)
            count += 1

    # Step 2: Fix the input/output of the graph now that we deleted
    # some args.
    gm.graph.lint()
    names = [f"arg_{i}" for i in range(len(in_spec.children_specs))]
    gm.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )
    gm.recompile()

    # Step 3: Find state references in HigherOrderOps and recursively
    # fix them.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.cond:
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    true_gm.register_buffer(operand.target, state_dict[operand.target])
                    false_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (pred, true_graph, false_graph, real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(true_gm, inp_pos_to_buffer_name_for_submod, in_spec, None, state_dict)
            _unlift(false_gm, inp_pos_to_buffer_name_for_submod, in_spec, None, state_dict)
        if node.op == "call_function" and node.target.__name__ == "map_impl":
            body_graph, num_mapped, *operands = node.args
            body_gm = getattr(gm, body_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    body_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (body_graph, num_mapped, *real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(body_gm, inp_pos_to_buffer_name_for_submod, in_spec, None, state_dict)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def unlift_exported_program_lifted_states(ep: torch._export.exported_program.ExportedProgram):
    new_gm = copy.deepcopy(ep.graph_module)

    for name, stuff in ep.state_dict.items():
        if name in ep.graph_signature.buffers:
            new_gm.register_buffer(name, stuff)
        elif name in ep.graph_signature.parameters:
            new_gm.register_parameter(name, stuff)
        else:
            raise AssertionError("encountered not registered param/buffer")

    count = 0
    inp_pos_to_buffer_name = {}
    for node in new_gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in ep.graph_signature.inputs_to_buffers:
                inp_pos_to_buffer_name[count] = ep.graph_signature.inputs_to_buffers[node.name]
            count += 1
    new_gm = _unlift(new_gm, inp_pos_to_buffer_name, ep.call_spec.in_spec, ep.call_spec.out_spec, ep.state_dict)
    return new_gm
