import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
    ConstantArgument,
    ModuleCallSignature,
    SymIntArgument,
    TensorArgument,
)
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(
    from_obj: torch.Tensor,
    to_module: torch.nn.Module,
    target: str,
    is_parameter: bool,
):
    *prefix, field = target.split(".")
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if not isinstance(from_obj, torch.Tensor):
        raise ValueError("Expected only parameters or buffers, got:", type(from_obj))

    if is_parameter:
        to_module.register_parameter(field, torch.nn.Parameter(from_obj))
    else:
        to_module.register_buffer(field, from_obj)


class _UnflattenedModule(torch.fx.GraphModule):
    def __init__(self, export_module: ExportedProgram):
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError("Unflattening on JointExportModule NYI")
        super().__init__({}, torch.fx.Graph(), "_UnflattenedModule")

        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        _inplace_buffer_mutations(export_graph, self.graph_signature)
        _outline_submodules(export_graph, self)

        self.range_constraints = export_module.range_constraints
        self.equality_constraints = export_module.equality_constraints

        state_dict = export_module.state_dict
        for name in self.graph_signature.parameters:
            cloned = state_dict[name].clone()
            _assign_attr(
                cloned,
                self,
                name,
                is_parameter=True,
            )
        for name in self.graph_signature.buffers:
            cloned = state_dict[name].clone()
            _assign_attr(
                cloned,
                self,
                name,
                is_parameter=False,
            )

        inputs_to_state: Dict[str, str] = {
            **self.graph_signature.inputs_to_parameters,
            **self.graph_signature.inputs_to_buffers,
        }

        _sink_params(self, inputs_to_state, [])
        # Check all input nodes has been processed.
        for module in self.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "placeholder":
                    continue
                assert node.name not in inputs_to_state

    def __call__(self, *args, **kwargs):
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        assert self.module_call_graph[0].fqn == ""
        signature = self.module_call_graph[0].signature
        if in_spec != signature.in_spec:
            raise TypeError(
                f"Input treespec does not match with exported module's. "
                "Are you sure you are calling this with the right arguments? "
                f"Input treespec: {in_spec}. ",
                f"Exported module treespec: {signature.in_spec}",
            )

        # TODO(zhxchen17) Use lineno map to dump the original stacktrace during error handling.
        tree_out = super().__call__(*flat_args)
        return pytree.tree_unflatten(tree_out, signature.out_spec)


def unflatten(module: ExportedProgram) -> _UnflattenedModule:
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module.
    """
    module = _UnflattenedModule(module)
    module.register_forward_pre_hook(_check_input_constraints_pre_hook)
    return module


def _inplace_buffer_mutations(graph: torch.fx.Graph, graph_signature) -> None:
    """Transform buffer mutations from their functionalized form into a copy_
    node in the graph.

    Functionalization represents buffer mutation by passing the buffer as an input and output. So for example, the eager code:
        def forward(self, x):
            self.buffer += x
            return x * x

    Will become a graph that looks like:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            mul = aten.mul(x, x)
            return (mutated_buffer, mul)

    We want to inplace this into something that looks like the original eager code:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            buffer.copy_(mutated_buffer)
            mul = aten.mul(x, x)
            return (mul,)
    """
    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output" and len(output_node.args) == 1
    return_args = output_node.args[0]

    mutation_node_to_buffer = graph_signature.buffers_to_mutate
    mutations = return_args[: len(mutation_node_to_buffer)]
    buffers_to_inputs = {v: k for k, v in graph_signature.inputs_to_buffers.items()}
    input_name_to_node = {
        node.name: node for node in graph.nodes if node.op == "placeholder"
    }

    for mutation in mutations:
        buffer_name = mutation_node_to_buffer[mutation.name]
        input_name = buffers_to_inputs[buffer_name]
        input_node = input_name_to_node[input_name]

        with graph.inserting_after(mutation):
            new_node = graph.create_node(
                "call_function", torch.ops.aten.copy_, (input_node, mutation)
            )
            for k, v in mutation.meta.items():
                new_node.meta[k] = v
        # Replace all uses of the previously functional mutation with our copy_ output.
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)

    # Remove the mutated buffer from the graph outputs, since we don't need to
    # thread it through anymore. We don't need to handle the inputs, which will
    # be handled by _sink_params.
    user_outputs = tuple(
        return_args[len(mutation_node_to_buffer) :],
    )
    output_node.args = ((user_outputs),)


def is_prefix(candidate, target):
    """Check whether `candidate` is a prefix of `target`."""
    return len(candidate) < len(target) and target[: len(candidate)] == candidate


def compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if parent_fqn == "":
        # Handle the root module correctly.
        return child_fqn

    parent_split = parent_fqn.split(".")
    child_split = child_fqn.split(".")

    assert (
        child_split[: len(parent_split)] == parent_split
    ), f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'"
    return ".".join(child_split[len(parent_split) :])


def _verify_graph_equivalence(x: torch.fx.GraphModule, y: torch.fx.GraphModule):
    def graph_dump(graph: torch.fx.Graph) -> str:
        ret = []
        nodes_idx: Dict[int, int] = {}

        def arg_dump(arg) -> str:
            if isinstance(arg, torch.fx.Node):
                return "%" + str(nodes_idx[id(arg)])
            return str(arg)

        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [
                f"{key}={value}"
                for key, value in pytree.tree_map(arg_dump, node.kwargs).items()
            ]
            target = node.target if node.op == "call_function" else ""
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return "\n".join(ret)

    assert graph_dump(x.graph) == graph_dump(y.graph)


def _add_spec(gm: torch.fx.GraphModule, spec) -> str:
    i = 0
    while hasattr(gm, f"_spec_{i}"):
        i += 1
    name = f"_spec_{i}"
    setattr(gm, name, spec)
    return name


def _generate_flatten(gm: torch.fx.GraphModule, node, spec) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


def _generate_unflatten(gm: torch.fx.GraphModule, nodes, spec) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


class ModuleFrame:
    def __init__(
        self,
        flat_graph,
        seen_nodes,
        seen_modules,
        parent,
        module_stack,
        module_id,
        module_call_graph: Dict[str, ModuleCallSignature],
        graph_module=None,
    ):
        self.flat_graph = flat_graph
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id

        self.module_call_graph = module_call_graph
        self.verbose = False

        self.fqn = self.module_stack[-1]
        if graph_module is not None:
            self.graph_module = graph_module
        else:
            # InterpreterModule doesn't work with torch.compile:
            # 1. in-place compile: nn.Module compile the forward function, and if we overwrite __call__,
            # in-place compile will not be effective
            # 2. out-of-place compile: there are a lot of graph guard failures on "self" in the
            # InterpreterModule
            # self.graph_module = InterpreterModule(
            self.graph_module = torch.fx.GraphModule(
                {},
                torch.fx.Graph(),
                self.fqn,
            )
            self.graph_module.meta["module_call_signature"] = module_call_graph.get(
                self.fqn
            )

        if self.module_id in self.seen_modules:
            self.cached_graph_module = self.seen_modules[self.module_id]
        else:
            self.cached_graph_module = None
            self.seen_modules[self.module_id] = self.graph_module

        self.nodes = list(self.flat_graph.nodes)
        self.graph = self.graph_module.graph

        # Mapping of nodes in the flat graph to nodes in this graph.
        self.node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}

        self.parent_call_module: Optional[torch.fx.Node] = None
        if parent is not None:
            accessor = compute_accessor(parent.fqn, self.fqn)
            parent.graph_module.add_submodule(
                accessor,
                self.graph_module
                if self.cached_graph_module is None
                else self.cached_graph_module,
            )
            self.parent_call_module = parent.graph.call_module(accessor)

        signature = module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            assert len(signature.in_spec.children_specs) == 2
            args_spec = signature.in_spec.children_specs[0]
            kwargs_spec = signature.in_spec.children_specs[1]
            assert args_spec.context is None
            assert kwargs_spec.context is not None

            with self.graph_module.graph.inserting_after(None):
                arg_nodes = []
                for idx in range(len(args_spec.children_specs)):
                    arg_nodes.append(
                        self.graph_module.graph.placeholder(f"_positional_arg_{idx}")
                    )
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph_module.graph.placeholder(name)
                flat_args = _generate_flatten(
                    self.graph_module,
                    (tuple(arg_nodes), kwarg_nodes),
                    signature.in_spec,
                )
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph_module.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(flat_args, idx),
                        name=arg.name
                        if not isinstance(arg, ConstantArgument)
                        else f"_constant_{idx}",
                    )
                    if isinstance(arg, ConstantArgument):
                        continue
                    flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                    self.node_to_placeholder[self.seen_nodes[arg.name]] = flat_arg_node

            with self.parent.graph.inserting_before(self.parent_call_module):
                nodes: List[Optional[torch.fx.Node]] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument) and input.value is None:
                        nodes.append(None)
                    else:
                        assert isinstance(input, (TensorArgument, SymIntArgument))
                        nodes.append(
                            self.parent.remap_input(self.seen_nodes[input.name])
                        )

                inputs_node = _generate_unflatten(
                    self.parent.graph_module,
                    nodes,
                    signature.in_spec,
                )

                args_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 0)
                )
                kwargs_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 1)
                )
                arg_nodes = [
                    self.parent.graph.call_function(operator.getitem, (args_node, i))
                    for i in range(len(args_spec.children_specs))
                ]
                kwarg_nodes = {
                    k: self.parent.graph.call_function(
                        operator.getitem, (kwargs_node, k)
                    )
                    for k in kwargs_spec.context
                }
            assert self.parent_call_module is not None
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes

    def add_placeholder(self, x):
        assert x.graph is self.flat_graph
        # x is not in subgraph, create a new placeholder for subgraph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # copy all meta fields, even if some fields might be irrelvant for
        # the placeholder node
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def remap_input(self, x):
        assert x.graph is self.flat_graph
        if x in self.node_map:
            return self.node_map[x]
        if x not in self.node_to_placeholder:
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                # Important to *prepend* the output to match how we are
                # inserting placeholder nodes.
                self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
        return self.node_to_placeholder[x]

    def finalize_outputs(self):
        orig_outputs = []

        signature = self.module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(output, (TensorArgument, SymIntArgument)):
                    orig_outputs.append(self.seen_nodes[output.name])
                else:
                    raise RuntimeError(
                        f"Unsupported data type for output node: {output}"
                    )

            tree_out_node = _generate_unflatten(
                self.graph_module,
                tuple(
                    self.node_map[self.seen_nodes[output.name]]
                    for output in orig_outputs
                ),
                signature.out_spec,
            )
            parent_out: Optional[torch.fx.Node] = _generate_flatten(
                self.parent.graph_module, self.parent_call_module, signature.out_spec
            )
            graph_outputs: Union[torch.fx.Node, List[torch.fx.Node]] = tree_out_node
        else:
            graph_outputs = []
            # Iterate through nodes we have copied into self.graph.
            for orig_node in self.node_map.keys():
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        # external user node, need to expose as an output
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break

            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]

        assert isinstance(graph_outputs, (list, torch.fx.Node))

        self.graph.output(graph_outputs)

        # lint to ensure correctness
        self.graph.lint()
        self.graph_module.recompile()

        # Rewrite outputs in parent module
        if parent_out is None:
            return

        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                # Use Proxy to record getitem access.
                proxy_out = torch.fx.Proxy(parent_out)[i].node  # type: ignore[index]
                self.parent.node_map[orig_output] = proxy_out

        if self.cached_graph_module is not None:
            _verify_graph_equivalence(self.cached_graph_module, self.graph_module)

    def copy_node(self, node):
        self.print("copying", node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        i = 0
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())
            i += 1

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run_from(self, node_idx):
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != "placeholder"

            self.print()
            self.print("STEP", node_idx, node.format_node())
            self.print(self.module_stack)
            if node.op == "output":
                if len(self.module_stack) == 1:
                    # We want the output node of the original graph to be handled
                    # specially by the outermost stack frame (in run_outer). So
                    # skip finalization here.
                    return node_idx

                # We've reached the end of the graph. Wrap up all the existing stack frames.
                self.finalize_outputs()
                return node_idx

            node_module_stack = (
                [path for path, ty in node.meta["nn_module_stack"].values()]
                if "nn_module_stack" in node.meta
                else self.module_stack
            )
            if node_module_stack[: len(self.module_stack)] != self.module_stack:
                # This means that the current module is done executing and the
                # current node is the beginning of a new module.
                #
                # In this case, we should finalize this module and return without
                # incrementing the node counter.
                self.finalize_outputs()
                self.print("outlining", self.fqn)
                self.print(self.graph)
                return node_idx

            assert node_module_stack is not None

            if is_prefix(self.module_stack, node_module_stack):
                # This means that the current node represents the execution of a new
                # module.
                next_module = node_module_stack[len(self.module_stack)]
                self.print("Creating new stack frame for", next_module)
                # Run a nested version of module outliner from the current node
                # counter. Once it is complete, continue from that point.
                node_idx = ModuleFrame(
                    self.flat_graph,
                    self.seen_nodes,
                    self.seen_modules,
                    self,
                    self.module_stack + [next_module],
                    list(node.meta["nn_module_stack"].keys())[len(self.module_stack)],
                    self.module_call_graph,
                ).run_from(node_idx)
                module_idx += 1
                continue

            # The only remaining possibility is that we are in the right stack
            # frame. Copy the node into this frame's graph and increment the node counter.
            assert node_module_stack == self.module_stack
            self.copy_node(node)
            node_idx += 1


def _outline_submodules(orig_graph: torch.fx.Graph, root_module: torch.fx.GraphModule):
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    ModuleFrame(
        orig_graph,
        seen_nodes,
        seen_modules,
        None,
        [""],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        graph_module=root_module,
    ).run_outer()


def _sink_params(
    module: GraphModule,
    inputs_to_state: Dict[str, str],
    scope: List[str],
):
    """Sink params and buffers from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containining nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    """
    for name, submodule in module._modules.items():
        _sink_params(cast(GraphModule, submodule), inputs_to_state, scope + [name])

    if not isinstance(module, GraphModule):
        return

    graph = module.graph
    inputs = filter(lambda n: n.op == "placeholder", graph.nodes)

    # Also remove from call_module nodes
    call_module_nodes = filter(lambda n: n.op == "call_module", graph.nodes)
    for node in call_module_nodes:
        node.args = tuple(filter(lambda n: n.name not in inputs_to_state, node.args))

    for node in inputs:
        if node.name not in inputs_to_state:
            continue

        if len(node.users) > 0:
            state_name = inputs_to_state[node.name].split(".")
            # If there's a mismatch beteewn scope name and state name, then there must be multuple scopes
            # pointing to the same state name, meaning some modules are shared. In such case, we can simply
            # skip updating the current node because another later iteration will take care of this input
            # node when the unique match between scope and state name occurs.
            # To make sure this always happen, we should enforce the invariant that no placeholder node
            # in the unflattened graph appears in inputs_to_state dict, which means all the extra input
            # nodes have been handled.
            if state_name[: len(scope)] != scope:
                continue
            attr_path = state_name[len(scope) :]
            state_attr = _recursive_getattr(module, attr_path)
            assert isinstance(state_attr, torch.Tensor)

            with graph.inserting_after(node):
                new_node = graph.create_node("get_attr", ".".join(attr_path))

            node.replace_all_uses_with(new_node, propagate_meta=True)
        graph.erase_node(node)
    module.recompile()


def _recursive_getattr(obj, attr_path):
    for attr in attr_path:
        obj = getattr(obj, attr)

    return obj
