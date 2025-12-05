"""
This module contains utility functions for working with joint FX graphs with descriptors
that are produced by AOTAutograd.  They will NOT work on generic FX graphs.  See also
:func:`torch._functorch.aot_autograd.aot_export_joint_with_descriptors`.  We also
recommend reading :mod:torch._functorch._aot_autograd.descriptors`.
"""

from typing import NoReturn, Optional, Union

import torch.fx as fx

from .descriptors import (
    AOTInput,
    AOTOutput,
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


def _raise_autograd_subclass_not_implemented(
    n: fx.Node, desc: Union[AOTInput, AOTOutput]
) -> NoReturn:
    raise RuntimeError(
        "Subclasses are currently not supported by this function, but a desugared subclass input "
        f"was found at {n} ({desc}).  The problem is "
        "that there may not necessarily be a 1-1 correspondence between primals/tangents/outputs/grads "
        "when subclasses are involved: for example, the primal might be a plain tensor "
        "but the tangent a tensor subclass that desugared into multiple plain tensors. "
        "It is not clear what exactly you would like this function to do in this case "
        "(Collect all nodes for the subclass together?  Match up the inner nodes if "
        "subclasses match exactly?)  If you have a concrete use case, please file an "
        "issue so we can understand it and design an API that works for your case."
    )


def get_all_input_and_grad_nodes(
    g: fx.Graph,
) -> dict[DifferentiableAOTInput, tuple[fx.Node, Optional[fx.Node]]]:
    """
    Given a joint graph with descriptors (meta['desc'] on placeholders and
    output), returns the node for every input and its corresponding grad
    output node if it exists.  These tuples are in a dict that is indexed by
    the AOTInput descriptor that describes the input.

    NB: *all* forward tensor inputs are returned, including non-differentiable
    inputs (which simply have a None grad), so it is safe to use this function
    to perform operations on all inputs.  (Non-tensor inputs like symbolic
    integers, tokens or RNG state are NOT traversed by this function.)

    Args:
        g: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each DifferentiableAOTInput descriptor to a tuple
        containing:
        - The input node itself
        - The grad (output) node if it exists, None otherwise

    Raises:
        RuntimeError: If the joint graph has subclass tensor inputs/outputs; this
        is not supported by API as there is not necessarily a 1-1 correspondence
        between inputs and grads when subclasses are involved.
    """
    input_index: dict[DifferentiableAOTInput, tuple[fx.Node, Optional[fx.Node]]] = {}
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            # Skip inputs that cannot possibly be differentiable
            if not isinstance(desc, DifferentiableAOTInput):
                continue
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_autograd_subclass_not_implemented(n, desc)
            # pyrefly: ignore [unsupported-operation]
            input_index[desc] = (n, None)
        elif n.op == "output":
            assert "desc" in n.meta, (n, n.meta)
            desc = n.meta["desc"]
            for sub_n, sub_desc in zip(n.args[0], desc):
                if isinstance(sub_desc, SubclassGetAttrAOTOutput):
                    _raise_autograd_subclass_not_implemented(sub_n, sub_desc)
                if isinstance(sub_desc, GradAOTOutput):
                    inp, grad = input_index[sub_desc.grad_of]
                    assert grad is None, (sub_n, sub_desc, input_index)
                    input_index[sub_desc.grad_of] = (inp, sub_n)
    return input_index


def get_all_output_and_tangent_nodes(
    g: fx.Graph,
) -> dict[DifferentiableAOTOutput, tuple[fx.Node, Optional[fx.Node]]]:
    """Get all output nodes and their corresponding tangent nodes from a joint graph.

    Similar to get_all_input_and_grad_nodes, but returns output nodes paired with
    their tangent nodes (if they exist). This function traverses the graph to find
    all differentiable outputs and matches them with their corresponding tangent
    inputs used in forward-mode autodiff.

    NB: *all* forward tensor output sare turned, including non-differentiable outputs,
    so you can use this function to perform operations on all outputs.

    Args:
        g: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each DifferentiableAOTOutput descriptor to a tuple
        containing:
        - The output node itself
        - The tangent (input) node if it exists, None otherwise

    Raises:
        RuntimeError: If the joint graph has subclass tensor inputs/outputs; this
        is not supported by API as there is not necessarily a 1-1 correspondence
        between outputs and tangents when subclasses are involved.
    """
    output_index: dict[DifferentiableAOTOutput, tuple[fx.Node, Optional[fx.Node]]] = {}
    for n in g.nodes:
        if n.op == "output":
            desc = n.meta["desc"]
            for sub_n, sub_d in zip(n.args[0], desc):
                # Skip outputs that cannot possibly be differentiable
                if not isinstance(sub_d, DifferentiableAOTOutput):
                    continue
                if isinstance(sub_d, SubclassGetAttrAOTOutput):
                    _raise_autograd_subclass_not_implemented(sub_n, sub_d)
                # pyrefly: ignore [unsupported-operation]
                output_index[sub_d] = (sub_n, None)
    for n in g.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_autograd_subclass_not_implemented(n, desc)
            if isinstance(desc, TangentAOTInput):
                out, tangent = output_index[desc.output]
                assert tangent is None, (n, desc, output_index)
                output_index[desc.output] = (out, n)
    return output_index


def get_param_and_grad_nodes(
    graph: fx.Graph,
) -> dict[ParamAOTInput, tuple[fx.Node, Optional[fx.Node]]]:
    """Get parameter nodes and their corresponding gradient nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each ParamAOTInput descriptor to a tuple containing:
        - The parameter input node
        - The gradient (output) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes(graph).items()
        if isinstance(desc, ParamAOTInput)
    }


def get_plain_input_and_grad_nodes(
    graph: fx.Graph,
) -> dict[PlainAOTInput, tuple[fx.Node, Optional[fx.Node]]]:
    """Get plain input nodes and their corresponding gradient nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each PlainAOTInput descriptor to a tuple containing:
        - The plain input node
        - The gradient (output) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_input_and_grad_nodes(graph).items()
        if isinstance(desc, PlainAOTInput)
    }


def get_plain_output_and_tangent_nodes(
    graph: fx.Graph,
) -> dict[PlainAOTOutput, tuple[fx.Node, Optional[fx.Node]]]:
    """Get plain output nodes and their corresponding tangent nodes from a joint graph.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping each PlainAOTOutput descriptor to a tuple containing:
        - The plain output node
        - The tangent (input) node if it exists, None otherwise
    """
    return {
        desc: (n, g)
        for desc, (n, g) in get_all_output_and_tangent_nodes(graph).items()
        if isinstance(desc, PlainAOTOutput)
    }


def _raise_fqn_subclass_not_implemented(
    n: fx.Node, desc: Union[AOTInput, AOTOutput]
) -> NoReturn:
    raise RuntimeError(
        "Subclasses are currently not supported by this function, but a desugared subclass input "
        f"was found at {n} ({desc}).  The problem is "
        "that there may not necessarily be a 1-1 correspondence between a FQN and a plain tensor "
        "when subclasses are involved: for example, a parameter that is a subclass "
        "would desugar into multiple plain tensors, which we can't uniquely assign the "
        "FQN to.  It's not clear what you want the API to do in this case: do you want to "
        "instead return a struct of nodes showing how to assemble the subclass?  But you "
        "don't (directly) have the metadata for the subclass?  If you have a concrete use "
        "case, please file an issue so we can understand it and design an API that works for your case."
    )


def get_named_param_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    """Get parameter nodes mapped by their fully qualified names.

    This function traverses the graph to find all parameter input nodes and
    returns them in a dictionary where keys are the parameter names (FQNs)
    and values are the corresponding FX nodes.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping parameter names (str) to their corresponding FX nodes.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        with subclasses a FQN does not necessarily map to a single plain tensor.
    """
    r = {}
    for n in graph.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_fqn_subclass_not_implemented(n, desc)
            elif isinstance(desc, ParamAOTInput):
                r[desc.target] = n
    return r


def get_named_buffer_nodes(graph: fx.Graph) -> dict[str, fx.Node]:
    """Get buffer nodes mapped by their fully qualified names.

    This function traverses the graph to find all buffer input nodes and
    returns them in a dictionary where keys are the buffer names (FQNs)
    and values are the corresponding FX nodes.

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A dictionary mapping buffer names (str) to their corresponding FX nodes.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        with subclasses a FQN does not necessarily map to a single plain tensor.
    """
    r = {}
    for n in graph.nodes:
        if n.op == "placeholder":
            desc = n.meta["desc"]
            if isinstance(desc, SubclassGetAttrAOTInput):
                _raise_fqn_subclass_not_implemented(n, desc)
            elif isinstance(desc, BufferAOTInput):
                r[desc.target] = n
    return r


def get_param_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Get all parameter nodes from a graph as a list.

    You can rely on this providing the correct order of parameters you need
    to feed into the joint graph (at the very beginning of the argument list,
    before buffers).

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A list of FX nodes representing all parameters in the graph.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        it is not clear if you wanted each individual constituent piece of the
        subclasses, or have them grouped up in some way.
    """
    return list(get_named_param_nodes(graph).values())


def get_buffer_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Get all buffer nodes from a graph as a list.

    You can rely on this providing the correct order of buffers you need
    to feed into the joint graph (after parameters).

    Args:
        graph: The FX joint graph with descriptors

    Returns:
        A list of FX nodes representing all buffers in the graph.

    Raises:
        RuntimeError: If subclass tensors are encountered (not yet supported), as
        it is not clear if you wanted each individual constituent piece of the
        subclasses, or have them grouped up in some way.
    """
    return list(get_named_buffer_nodes(graph).values())
