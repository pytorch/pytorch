from collections.abc import Sequence
from typing import Optional

import torch.fx as fx

from .descriptors import (
    BufferAOTInput,
    DifferentiableAOTInput,
    DifferentiableAOTOutput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    TangentAOTInput,
)


def _raise_subclass_not_implemented():
    raise RuntimeError(
        "Subclasses currently not supported by this function.  The problem is "
        "that there may not necessarily be a 1-1 correspondence between primals/tangents/outputs/grads "
        "when subclasses are involved: for example, the primal might be a plain tensor "
        "but the tangent a tensor subclass that desugared into multiple plain tensors. "
        "It is not clear what exactly you would like this function to do in this case "
        "(Collect all nodes for the subclass together?  Match up the inner nodes if "
        "subclasses match exactly?)  If you have a concrete use case, please file an "
        "issue so we can understand it and design an API that works for your case."
    )


def get_all_input_and_grad_nodes_indexed(
    g: fx.Graph,
) -> dict[DifferentiableAOTInput, tuple[fx.Node, Optional[fx.Node]]]:
    """
    Given a joint graph with descriptors (meta['desc'] on placeholders and
    output), returns a zipped sequence of the node for every input, and its
    corresponding grad output node if it exists.  NB: *all* forward tensor
    inputs are returned, including non-differentiable inputs (which simply
    have a None grad), so it is safe to use this function to perform
    operations on all inputs.  (Non-tensor inputs like symbolic integers,
    tokens or RNG state are NOT traversed by this function.)

    If AOTAutograd ever supports double backwards, this will return ALL input
    and grad output pairs, which will imply that backward inputs can show up
    may also show up in the outputs here (e.g., tangents and grad of tangents).
    """
    input_index: dict[DifferentiableAOTInput, tuple[fx.Node, Optional[fx.Node]]] = {}
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            # Skip inputs that cannot possibly be differentiable
            if not isinstance(desc, DifferentiableAOTInput):
                continue
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_subclass_not_implemented()
            input_index[desc] = (n, None)
        elif n.op == "output":
            assert "desc" in n.meta, (n, n.meta)
            desc = n.meta["desc"]
            for sub_n, sub_desc in zip(n.args[0], desc):
                if isinstance(sub_desc, SubclassGetAttrAOTOutput):
                    _raise_subclass_not_implemented()
                if isinstance(sub_desc, GradAOTOutput):
                    inp, grad = input_index[sub_desc.grad_of]
                    assert grad is None, (sub_n, sub_desc, input_index)
                    input_index[sub_desc.grad_of] = (inp, sub_n)
    return input_index


def get_all_input_and_grad_nodes(
    g: fx.Graph,
) -> Sequence[tuple[fx.Node, Optional[fx.Node]]]:
    return get_all_input_and_grad_nodes_indexed(g).values()


def get_all_output_and_tangent_nodes_indexed(
    g: fx.Graph,
) -> dict[DifferentiableAOTOutput, tuple[fx.Node, Optional[fx.Node]]]:
    """Like get_all_input_and_grad_nodes, but for outputs and their tangent nodes."""
    output_index: dict[DifferentiableAOTOutput, tuple[fx.Node, Optional[fx.Node]]] = {}
    for n in g.nodes:
        if n.op == "output":
            desc = n.meta["desc"]
            for sub_n, sub_d in zip(n.args[0], desc):
                # Skip outputs that cannot possibly be differentiable
                if not isinstance(sub_d, DifferentiableAOTOutput):
                    continue
                if isinstance(sub_d, SubclassGetAttrAOTOutput):
                    _raise_subclass_not_implemented()
                output_index[sub_d] = (sub_n, None)
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_subclass_not_implemented()
            if isinstance(desc, TangentAOTInput):
                out, tangent = output_index[desc.output]
                assert tangent is None, (n, desc, output_index)
                output_index[desc.output] = (out, n)
    return output_index


def get_all_output_and_tangent_nodes(
    g: fx.Graph,
) -> Sequence[tuple[fx.Node, Optional[fx.Node]]]:
    return get_all_output_and_tangent_nodes(g).values()


# TODO: It's not clear to me if you actually want these two separate variants?
# Autoparallel seems to do slightly different things for these cases but it's
# not clear that it is really right to do so.


def get_param_and_grad_nodes(
    graph: fx.Graph,
) -> Sequence[tuple[fx.Node, Optional[fx.Node]]]:
    """Like get_all_input_and_grad_nodes, but only returning parameter inputs.

    NB: Parameter here also includes buffers, which will typically have no grad node."""
    return [
        (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes_indexed(graph).items()
        if isinstance(desc, ParamAOTInput)
    ]


def get_plain_input_and_grad_nodes(
    graph: fx.Graph,
) -> Sequence[tuple[fx.Node, Optional[fx.Node]]]:
    """Like get_all_input_and_grad_nodes, but only returning plain (non-parameter) inputs."""
    return [
        (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes_indexed(graph).items()
        if isinstance(desc, PlainAOTInput)
    ]


def get_plain_output_and_tangent_nodes_indexed(
    graph: fx.Graph,
) -> dict[PlainAOTOutput, tuple[fx.Node, Optional[fx.Node]]]:
    """Like get_all_input_and_grad_nodes, but only returning plain (non-parameter) inputs."""
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_output_and_tangent_nodes_indexed(graph).items()
        if isinstance(desc, PlainAOTOutput)
    }


# TODO: this one can support subclasses, although you have to make a decision
# if you're going to return things that are flattened inner pieces of
# subclasses
def get_param_nodes(graph: fx.Graph) -> Sequence[fx.Node]:
    """Return all ParamAOTInput nodes."""
    return [n for n, _ in get_param_and_grad_nodes(graph)]


def named_param_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    return {
        desc.target: n
        for desc, (n, _) in get_all_input_and_grad_nodes_indexed(graph).items()
        if isinstance(desc, ParamAOTInput)
    }


def named_buffer_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    return {
        desc.target: n
        for desc, (n, _) in get_all_input_and_grad_nodes_indexed(graph).items()
        if isinstance(desc, BufferAOTInput)
    }
