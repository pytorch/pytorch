# mypy: allow-untyped-defs
import builtins
import logging
import operator
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.export._trace
from torch import _C
from torch._export.passes.replace_quantized_ops_with_standard_ops_pass import (
    replace_quantized_ops_with_standard_ops,
)
from torch.export.dynamic_shapes import _tree_map_with_path, Dim
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx import subgraph_rewriter


log = logging.getLogger(__name__)


def _get_param_count_list(method_graph, args_params):
    param_count_list = []
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        if "PackedParams" in str(input_.type()):
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            param_count_list.append(arg_params_ is not None)

    return param_count_list


def _trace_and_get_graph_from_model(model, args):
    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    # Disable Autocast cache because it replaces kernel's weight and bias
    # by (undesired) constants.
    # No perf impact for when there are reused weights since https://github.com/pytorch/pytorch/pull/85665
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    trace_graph, torch_out, _inputs_states = torch.jit._get_trace_graph(
        model,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)

    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _create_jit_graph(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction], args: Sequence[Any]
) -> tuple[torch.Graph, list["_C.IValue"], Any, Optional[torch.ScriptModule]]:
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        torch_out = None

        if isinstance(model, torch.jit.ScriptModule):
            try:
                graph = model.forward.graph  # type: ignore[attr-defined]
            except AttributeError as e:
                raise RuntimeError("'forward' method must be a script method") from e
            _C._jit_pass_onnx_function_substitution(graph)
            freezed_module = _C._freeze_module(
                typing.cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            method_graph = module._get_method("forward").graph
            args_params = tuple(args) + tuple(params)
            param_count_list = _get_param_count_list(method_graph, args_params)
            in_vars, _ = torch.jit._flatten(args_params)
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            return graph, params, torch_out, module

        # torch.jit.ScriptFunction
        params = []
        graph = model.graph
        _C._jit_pass_onnx_function_substitution(graph)
        param_count_list = _get_param_count_list(graph, args)
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        return graph, params, torch_out, None

    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    _C._jit_pass_onnx_lint(graph)
    state_dict = torch.jit._unique_state_dict(model)
    params = list(state_dict.values())
    graph_inputs = list(graph.inputs())
    user_input_num = len(graph_inputs) - len(state_dict)
    param_names = list(state_dict.keys())
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    _C._jit_pass_onnx_function_substitution(graph)
    return graph, params, torch_out, None


def list_add(a, b):
    return a + b


def list_append(container, element):
    return container + [element]


def execute_subgraph_from_prim_loop(
    subgraph, iter_idx, len_loop_local_arguments, *args, **kwargs
):
    """
    subgraph: GraphModule from sub-block.
    iter_idx: The index of interaction.
    len_loop_local_arguments: The number of loop local arguments in args.
    """

    # Loop local variables. TS graph create those as inputs because their values
    # are updated inside the loop.
    loop_local_args = args[:len_loop_local_arguments]
    # Global variables that are not passed in as inputs to the loop sub-blocks
    # but are directly used. Most of time, their values are not updated, but
    # the only exception is when there are some operations that perform inplace
    # updates.
    global_args = args[len_loop_local_arguments:]
    return subgraph(*global_args, iter_idx, *loop_local_args, **kwargs)


def inplace_optimize_sym_size_div(gm: torch.fx.GraphModule):
    def pattern(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        scalar_tensor = torch.ops.aten.scalar_tensor(sym_size_int)
        div_scalar_mode = torch.ops.aten.div.Scalar_mode(
            scalar_tensor, scale, rounding_mode="trunc"
        )
        int_tensor = torch.ops.aten.Int.Tensor(div_scalar_mode)
        return int_tensor

    def replacement(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        return sym_size_int // scale

    subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def is_valid_for_codegen(name):
    if len(name) == 0:
        raise RuntimeError("Empty argument name for codegen")
    if name[0].isdigit():
        return False
    return True


def normalize_name(name: str, prefix: str = "rename") -> str:
    name = name.replace(".", "_")
    if is_valid_for_codegen(name):
        return name
    return f"{prefix}_{name}"


def ir_name_to_func_name(name: str) -> str:
    """prim::If -> convert_prim_If"""
    name_list = name.split("::")
    return "convert_" + "_".join(name_list)


def get_node_as_placeholder_or_get_attr(fx_graph, name, is_top_level_graph):
    if is_top_level_graph:
        return fx_graph.get_attr(name)
    return fx_graph.placeholder(name)


_TORCH_DTYPE_TO_ENUM = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.qint8: 12,
    torch.quint8: 13,
    torch.bfloat16: 15,
}

_TORCH_ENUM_TO_DTYPE = {value: key for key, value in _TORCH_DTYPE_TO_ENUM.items()}


def get_dtype_as_int(tensor):
    """
    prim::dtype has the signature "Tensor a) -> int", where it gets the dtype of
    the tensor and returns the integer corresponding to this dtype based on the
    enum in ScalarType.h
    """
    dtype = tensor.dtype
    if dtype not in _TORCH_DTYPE_TO_ENUM:
        raise RuntimeError(f"Unsupported dtype {dtype}")
    return _TORCH_DTYPE_TO_ENUM[dtype]


# Those operators will be automatically populated to a instance method
# of TS2FXGraphConverter with name convert_<namespace>_<opname>().
# Please check __init__ for method population implementations.
kind_to_standard_operators: dict[str, Callable[..., Any]] = {
    "prim::max": builtins.max,
    "prim::min": builtins.min,
    "prim::TupleIndex": operator.getitem,
    "aten::__is__": operator.is_,
    "aten::__isnot__": operator.is_not,
    "aten::__not__": operator.not_,
    "aten::__contains__": operator.contains,
    "prim::dtype": get_dtype_as_int,
    "aten::len": len,
    # Mapping from specialized op to its symbolic counterpart.
    # They currently do not have any other overrides.
    "aten::numel": torch.ops.aten.sym_numel,
    "aten::size": torch.ops.aten.sym_size,
    "aten::storage_offset": torch.ops.aten.sym_storage_offset,
    "aten::stride": torch.ops.aten.sym_stride,
}


def get_ir_value_parent_name_and_attr_name(node):
    irv_parent_name, irv_name = node.input().debugName(), node.output().debugName()
    attr_name = node.s("name")
    return irv_name, irv_parent_name, attr_name


def construct_fqn(ir, ref_map, name_map):
    name_list = []
    while ir in ref_map:
        name_list.append(name_map[ir])
        ir = ref_map[ir]
    return ".".join(reversed(name_list))


def get_block_to_lifted_attrs(
    graph: torch._C.Graph,
) -> tuple[dict[torch._C.Block, set[str]], dict[str, str]]:
    """
    Perform two passes to get a mapping of blocks to a set of FQNs of its lifted attributes.
    When a graph has control flow, the graph will be divided into multiple blocks. We want to convert
    each block to a graph which will be passed into torch.cond. A restriction for torch.cond is that model
    parameters/buffers are expected to be lifted as inputs to the subgraphs. Before converting the model,
    we will run this pass which will:
        1. Figure out which params/buffers are used within blocks through tracing the GetAttr calls.
        2. Process the graph bottom up to find the lifted attributes of each block by taking the union
        of the attributes used in the current block, and the lifted attributes of all its child blocks.

    Returns:
        A mapping of blocks to a set of FQNs of its lifted attributes, and a
        mapping of node names to the FQNs of its lifted attributes.
    """

    # A map from a block to its expected to be lifted arguments.
    blocks_to_lifted_attrs: dict[torch._C.Block, set[str]] = {}

    # Reference map stores the input (i.e., src) and output (i.e., dest) IR of a
    # GetAttr node. By traversing this reference map, we can figure out the
    # full IR aliasing pass and figure out the FQN of an attribute.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_parent_map["%2"] = "%1"
    node_to_parent_map: dict[str, str] = {}

    # Used for reconstructing the FQN of an attribute based on the reference map.
    # In nutshell, for each GetAttr call, GetAttr(input IR, attribute name) -> output IR
    # This name map stores which attribute name is called for a src IR --> dest IR action.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_attr_name["%2"] = "linear"
    node_to_attr_name: dict[str, str] = {}

    def _dfs_get_attr_dependency(entry):
        """
        First DFS path to construct reference map and name map.
        """
        for node in entry.nodes():
            if node.kind() == "prim::GetAttr":
                (
                    irv_name,
                    irv_parent_name,
                    attr_name,
                ) = get_ir_value_parent_name_and_attr_name(node)
                node_to_parent_map[irv_name] = irv_parent_name
                node_to_attr_name[irv_name] = attr_name
            for block in node.blocks():
                _dfs_get_attr_dependency(block)

    def _map_blocks_to_lifted_attrs(entry):
        """
        Walk the graph in a bottom-up fashion to build the expected to be
        lifted arguments for each block.
        """
        arguments: set[str] = set()
        for node in entry.nodes():
            for block in node.blocks():
                # Recursively build.
                arguments = arguments.union(_map_blocks_to_lifted_attrs(block))
            if node.kind() == "prim::GetAttr":
                irv_name = node.output().debugName()
                # Skip for intermediate GetAttr, which will anyway not result a FQN.
                # E.g., node_to_parent_name: {"%3": "%2", "%2": "%1"}
                #       node_to_attr_name: {"%3": "weight", "%2": "linear", "%1": "self"}
                #       There is only one FQN %3-->%2-->%1: self.linear.weight
                #       %2-->%1 is not a FQN: self.linear
                if irv_name not in set(node_to_parent_map.values()):
                    arguments.add(
                        construct_fqn(irv_name, node_to_parent_map, node_to_attr_name)
                    )
        if not isinstance(entry, torch._C.Graph):  # Skip the top level.
            blocks_to_lifted_attrs[entry] = arguments
        return arguments

    _dfs_get_attr_dependency(graph)
    _map_blocks_to_lifted_attrs(graph)

    return blocks_to_lifted_attrs, node_to_attr_name


def get_attribute_fqn_from_ts_node(
    name_to_attribute_fqn: dict[str, str], node: torch._C.Node
) -> str:
    def get_attr(name: str):
        if name in name_to_attribute_fqn:
            return name_to_attribute_fqn[name]
        else:
            raise ValueError(f"Attribute {name} not found")

    if node.kind() == "prim::SetAttr":
        input_name = next(node.inputs()).debugName()
    elif node.kind() == "prim::GetAttr":
        input_name = node.input().debugName()
    else:
        raise RuntimeError(
            f"Unexpected node kind when getting attribute fqn. node: {node} "
        )

    attr_name = node.s("name")
    root_attr_name = get_attr(input_name)
    attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name

    return attr_fqn


def get_op_overload(node: torch._C.Node):
    schema_str = node.schema()
    assert schema_str != "(no schema)", f"got empty schema for {node}"
    schema: torch._C.FunctionSchema = torch._C.parse_schema(schema_str)
    ns, op_name = str(schema.name).split("::")
    override = schema.overload_name

    try:
        op_overload_mod = getattr(torch.ops, ns)
        op_overload_packet = getattr(op_overload_mod, op_name)
        if override:
            op_overload = getattr(op_overload_packet, override)
        else:
            op_overload = op_overload_packet.default
    except Exception as e:
        raise RuntimeError(
            f"Unable to find operator {node.kind()} with schema {node.schema()}"
        ) from e

    return op_overload


class TS2FXGraphConverter:
    def __init__(
        self,
        ts_graph: Union[torch._C.Graph, torch._C.Block],
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        self.ts_graph = ts_graph
        # Mapping of parameter FQN to actual parameter value
        self.name_to_param = name_to_param
        # Mapping of buffer FQN to actual buffer value
        self.name_to_buffer = name_to_buffer

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: list[InputSpec] = []
        self.output_specs: list[OutputSpec] = []

        # Mapping of TS node name to converted FX node
        self.name_to_node: dict[
            str, Union[torch.fx.Node, list[torch.fx.Node], dict[Any, torch.fx.Node]]
        ] = {}
        # Mapping of TS node name to constant value (int, str, TorchBind obj,
        # tensor constants ...)
        self.name_to_constant: dict[str, Any] = name_to_constant

        # Mapping from torchscript node output name to attribute fully qualified name
        self.name_to_attribute_fqn: dict[str, str] = name_to_attribute_fqn

        # Mapping from fully qualified name to real values or a fx graph node
        # During convert, this represents the current value of a non-tensor attribute
        # One use case is:
        #   def forward(self, x):
        #        c1 = self.count
        #        self.count += 1
        #        c2 = self.count
        #        return x + c1 + c2
        self.name_to_non_tensor_attribute_node: dict[str, Any] = {}

        # Mapping from fully qualified name to initial real values inputs
        # We separate it from self.name_to_non_tensor_attribute_node since
        # we need initial real value input when we construct fx.GraphModule
        self.name_to_non_tensor_attribute: dict[str, Any] = name_to_non_tensor_attribute

        self.subgraphs: dict[str, torch.fx.GraphModule] = {}

        # Mapping of block to list of attributes that need to be lifted for each
        # block
        self.blocks_to_lifted_attrs = blocks_to_lifted_attrs

        # Populate methods for the standard operators.
        for k in kind_to_standard_operators:
            handler_func_name = ir_name_to_func_name(k)
            # Create an indirect function call:
            # convert_<namespace>_<opname> --> lambda node: _convert_standard_operator(node)
            setattr(
                self,
                handler_func_name,
                lambda node: self._convert_standard_operators(node),
            )

        # This stores a list of return results that do not appear in the original TS
        # graph's outputs. The reason we maintain this is because some operations in the sub-block
        # might have inplace updates to the variable defined in the parent fx graph. After
        # the execution of that sub-block, the variable defined in the parent fx graph also
        # needs to be updated.
        self.name_update_from_subblock_to_parent: set[str] = set()

    def _is_get_attr_node(self, fqn):
        return (
            fqn in self.name_to_buffer
            or fqn in self.name_to_param
            or (
                fqn in self.name_to_constant
                and isinstance(self.name_to_constant[fqn], torch.ScriptObject)
            )
        )

    def _convert_block_to_subgraph(self, node: torch._C.Node, arguments: list[str]):
        subgraph_nodes, subgraph_converters = [], []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(
                block,
                self.name_to_param,
                self.name_to_buffer,
                self.blocks_to_lifted_attrs,
                {},
                self.name_to_constant,
                self.name_to_attribute_fqn,
            )

            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))
            subgraph_converters.append(subgraph_converter)
        return subgraph_nodes, subgraph_converters

    def _identify_inputs_as_arguments(self, entry):
        """
        Identify inputs from the innermost sub-block. This is needed
        for nested sub-blocks when the input is hidden in the nested sub-block.
        E.g., example IR of input is hidden in the nested sub-block.
        Graph[x.1]
        %1 = ...
            Block[]
                Block[x.1]
                    %2 = x.1 ...
        """
        arguments: set[str] = set()
        for block in entry.blocks():
            for block_node in block.nodes():
                for block_node_in in block_node.inputs():
                    if (
                        block_node_in.debugName() in self.name_to_node
                        and block_node_in.debugName() not in self.name_to_attribute_fqn
                    ):
                        arguments.add(block_node_in.debugName())
                arguments = arguments.union(
                    self._identify_inputs_as_arguments(block_node)
                )
        return arguments

    def is_top_level_graph(self):
        return isinstance(self.ts_graph, torch._C.Graph)

    def add_subgraph(self, subgraph) -> str:
        name = f"subgraph_{len(self.subgraphs)}"
        self.subgraphs[name] = subgraph
        return name

    def get_args_kwargs(self, node: torch._C.Node, schema):
        args = []
        kwargs = {}
        for input, schema_arg in zip(node.inputs(), schema.arguments):
            if schema_arg.kwarg_only:
                kwargs[schema_arg.name] = self.get_fx_value_by_ir_value(input)
            else:
                args.append(self.get_fx_value_by_ir_value(input))

        return tuple(args), kwargs

    def get_fx_value_by_ir_value(self, value: torch._C.Value):
        value_name = value.debugName()

        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        elif value_name in self.name_to_constant:
            if isinstance(self.name_to_constant[value_name], torch.ScriptObject):
                return self.fx_graph.get_attr(value_name)
            return self.name_to_constant[value_name]
        elif value_name in self.name_to_attribute_fqn:
            return self.get_fx_value_by_fqn(self.name_to_attribute_fqn[value_name])
        else:
            raise ValueError(f"Input {value_name} not found")

    def get_fx_value_by_fqn(self, name):
        if name in self.name_to_node:
            fx_node = self.name_to_node[name]
        elif name in self.name_to_constant:
            fx_node = self.name_to_constant[name]
        elif name in self.name_to_non_tensor_attribute_node:
            fx_node = self.name_to_non_tensor_attribute_node[name]
        elif name in self.name_to_non_tensor_attribute:
            fx_node = self.name_to_non_tensor_attribute[name]
        else:
            raise ValueError(f"Attribute {name} not found")
        return fx_node

    def convert(self) -> torch.fx.GraphModule:
        self.convert_graph_inputs()

        for node in self.ts_graph.nodes():
            self.convert_node(node)

        self.convert_graph_outputs()

        # Pass parameter and buffer to the root for lookup.
        gm = torch.fx.GraphModule(
            {
                **self.subgraphs,
                **self.name_to_param,
                **self.name_to_buffer,
                **self.name_to_non_tensor_attribute,
                **self.name_to_constant,
            },
            self.fx_graph,
        )

        inplace_optimize_sym_size_div(gm)

        gm.graph.lint()

        return gm

    def convert_graph_inputs(self):
        for graph_input in self.ts_graph.inputs():
            name = graph_input.debugName()

            if name in self.name_to_param:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_buffer:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_constant:
                assert isinstance(self.name_to_constant[name], torch.ScriptObject), (
                    "Input conversion only handles ScriptObject"
                )
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.CUSTOM_OBJ,
                        arg=CustomObjArgument(
                            name=normalized_name, class_fqn=normalized_name
                        ),
                        target=name,
                        persistent=False,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif isinstance(graph_input.type(), torch.ClassType):
                # Directly skip inputs that are ScriptObject but not used in the graph.
                continue
            else:
                normalized_name = normalize_name(name, prefix="input")
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = self.fx_graph.placeholder(normalized_name)

            self.name_to_node[name] = fx_node

    def convert_aten_Float(self, node: torch._C.Node):
        def to_float_tensor(t):
            return t.to(dtype=torch.float).item()

        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]  # noqa: C416
        fx_node = self.fx_graph.call_function(
            to_float_tensor,
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_aten_tensor(self, node: torch._C.Node):
        """aten::tensor creates a constant tensor ad-hoc --> GetAttr"""
        args, kwargs = self.get_args_kwargs(node, torch.ops.aten.tensor.default._schema)

        for k in kwargs:
            if k == "requires_grad":
                kwargs[k] = bool(kwargs[k])  # 0 -> False, 1 -> True

        to_tensor = (
            torch.tensor
            if all(isinstance(a, int) for a in args)
            else torch._refs.tensor
        )

        def target(*args, **kwargs):
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
            return to_tensor(*args, **kwargs)

        # def to_dynamic_tensor(*args, **kwargs):
        #     if "dtype" in kwargs and kwargs["dtype"] is not None:
        #         kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
        #     return torch._refs.tensor(*args, **kwargs)

        output_name = node.output().debugName()
        fx_node = self.fx_graph.call_function(target, args, kwargs)
        self.name_to_node[output_name] = fx_node

    def convert_aten_append(self, node: torch._C.Node):
        # special handle python list append: "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"

        # inplace append to the list!! This is kinda crazy, as we are inplace mutating the list
        # This makes the converter "non-functional", and the result depends on the order of the nodes being converter
        # In a sense, the converter now becomes an stateful interpreter
        warnings.warn(
            "Converting aten::append.t, which is a inplace mutation of the list. "
            "This makes the converter non-functional: the result depends on the order of the append nodes being converter!",
            stacklevel=2,
        )

        args = tuple(self.get_fx_value_by_ir_value(inp) for inp in node.inputs())
        fx_node = self.fx_graph.call_function(list_append, args)
        self.name_to_node[node.output().debugName()] = fx_node

        # inplace mutate arg[0], which is the python list
        self.name_to_node[node.inputsAt(0).debugName()] = fx_node

        # Variables that need to be updated to parent module.
        if not self.is_top_level_graph() and args[0].op == "placeholder":
            self.name_update_from_subblock_to_parent.add(node.inputsAt(0).debugName())

    def convert_prim_Constant(self, node: torch._C.Node):
        name = node.output().debugName()

        value: Any = None
        if node.hasAttribute("value"):
            constant_kind = node.kindOf("value")
            if constant_kind == "i":
                value = node.i("value")
            elif constant_kind == "f":
                value = node.f("value")
            elif constant_kind == "s":
                value = node.s("value")
            elif constant_kind == "t":
                alias_name = (
                    f"lifted_tensor_{name}"  # Follow naming convention from EP tracing.
                )
                fx_node = self.fx_graph.get_attr(alias_name)
                self.name_to_node[name] = fx_node
                name, value = alias_name, node.t("value")
            elif constant_kind == "ival":
                value = node.ival("value")
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")
        else:
            value = None

        self.name_to_constant[name] = value

    def convert_prim_CallMethod(self, node: torch._C.Node):
        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]  # noqa: C416
        fx_node = self.fx_graph.call_method(
            node.s("name"),
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_prim_device(self, node: torch._C.Node):
        input_type = node.input().type()
        if input_type.isSubtypeOf(torch._C.TensorType.get()):
            device = input_type.device()  # type: ignore[attr-defined]
            output_name = node.output().debugName()
            self.name_to_constant[output_name] = device
        else:
            raise ValueError(f"Unsupported JitType ({input_type}) when get device")

    def convert_prim_GetAttr(self, node: torch._C.Node):
        # Build fully qualified name
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = attr_fqn

        if self.is_top_level_graph():
            if self._is_get_attr_node(attr_fqn):
                # We insert a get_attr node due to two reasons.
                # First, ts graph does not lift tensor constants as input nodes. So
                # tensor constants may be ignored by in convert_graph_inputs().
                # Second, attr_fqn may have been written to via SetAttr. Two
                # GetAttr may give different values.
                self.name_to_node[output_name] = self.fx_graph.get_attr(attr_fqn)
            else:
                if attr_fqn not in self.name_to_non_tensor_attribute_node:
                    self.name_to_non_tensor_attribute_node[attr_fqn] = (
                        self.name_to_non_tensor_attribute[attr_fqn]
                    )
                self.name_to_node[output_name] = self.name_to_non_tensor_attribute_node[
                    attr_fqn
                ]
        else:
            # Special support for if blocks which do not allow SetAttr TorchScript
            # node and get_attr FX Graph Node.
            if self._is_get_attr_node(attr_fqn):
                self.name_to_node[output_name] = self.name_to_node[attr_fqn]

    def convert_prim_SetAttr(self, node: torch._C.Node):
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        attr_value = tuple(node.inputs())[1]
        ts_graph_tensor_input = self.get_fx_value_by_ir_value(attr_value)
        if self._is_get_attr_node(attr_fqn):
            fx_attr_node = self.fx_graph.get_attr(attr_fqn)
            self.fx_graph.call_function(
                torch.Tensor.copy_, (fx_attr_node, ts_graph_tensor_input)
            )
        else:
            self.name_to_non_tensor_attribute_node[attr_fqn] = ts_graph_tensor_input

    def convert_call_function_op(self, node: torch._C.Node):
        target = get_op_overload(node)

        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = fx_node
        else:
            for i, outp in enumerate(node.outputs()):
                output_name = outp.debugName()
                next_fx_node = self.fx_graph.call_function(
                    operator.getitem, (fx_node, i)
                )
                self.name_to_node[output_name] = next_fx_node

    def convert_prim_TupleConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def _convert_prim_iterator(self, node: torch._C.Node):
        output_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_list

    def convert_prim_DictConstruct(self, node: torch._C.Node):
        output_dict = {}
        k, v = None, None
        for i, inp in enumerate(node.inputs()):
            # We assume key value are stored in pair in the DictConstruct.
            # The first element is the key and the following is the value.
            if i % 2 == 0:
                k = self.get_fx_value_by_ir_value(inp)
            else:
                v = self.get_fx_value_by_ir_value(inp)
                assert k is not None and v is not None, (
                    "DictConstruct has an empty key value pair."
                )
                output_dict[k] = v
                k, v = None, None

        assert k is None and v is None, (
            "DictConstruct has an odd number of elements (violating our assumption)."
        )

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_dict

    def convert_prim_ListUnpack(self, node: torch._C.Node):
        self._convert_prim_unpack_iterator(node)

    def convert_prim_TupleUnpack(self, node: torch._C.Node):
        self._convert_prim_unpack_iterator(node)

    def _convert_prim_unpack_iterator(self, node: torch._C.Node):
        # Single input and multiple outputs for unpacking.
        for i, outp in enumerate(node.outputs()):
            outp_name = outp.debugName()
            inp = self.get_fx_value_by_ir_value(node.input())
            fx_node = self.fx_graph.call_function(operator.getitem, (inp, i))
            self.name_to_node[outp_name] = fx_node

    def convert_aten_Int(self, node: torch._C.Node):
        # converts aten::Int as aten._to_copy + aten::_local_scalar_dense
        target = torch.ops.aten._to_copy.default
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # Converts prim::NumToTensor as aten.scalar_tensor.
        # prim::NumToTensor IRs are currently triggered by:
        # .size() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L950
        # .numel() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L971
        # For both of those APIs, torch.jit.trace implicitly sets the output tensor type
        # to be LongTensor.
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())

        fx_node = self.fx_graph.call_function(target, args, {"dtype": torch.long})
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = ""

    def convert_aten__convolution(self, node: torch._C.Node):
        # converts aten::_convolution as aten.convolution, since aten::_convolution
        # doesn't have a meta function
        target = torch.ops.aten.convolution.default
        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_div(self, node: torch._C.Node):
        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        # converts aten::div.Tensor_mode(x, tensor_constant)
        # as aten.div.Scalar_mode(x, tensor_constant.item())
        if schema.overload_name == "Tensor_mode":
            arg1_name = args[1].name
            if arg1_name in self.name_to_constant and isinstance(
                self.name_to_constant[arg1_name], torch.Tensor
            ):
                tensor_constant = self.name_to_constant[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.name_to_constant[arg1_name].item()

                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: convert sourceRange() into stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    output_name = node.output().debugName()
                    self.name_to_node[output_name] = fx_node
                    return

        self.convert_call_function_op(node)

    def convert_aten___getitem__(self, node: torch._C.Node):
        input_container, index = tuple(
            self.get_fx_value_by_ir_value(input) for input in node.inputs()
        )
        fx_node = self.fx_graph.call_function(
            operator.getitem, (input_container, index)
        )
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_to(self, node: torch._C.Node):
        target = get_op_overload(node)
        args, _kwargs = self.get_args_kwargs(node, target._schema)

        # special handle aten.to.dtype and aten.to.prim_dtype followed by inplace_mutation_op
        # coz aten.to + inplace_mutation_op pattern would trigger
        # "cannot mutate tensors with frozen storage" functionalization error.
        # To work around the issue, we override the copy to be True, so that the output
        # is for sure not an alias of input
        if target is torch.ops.aten.to.dtype or target is torch.ops.aten.to.prim_dtype:
            user_nodes = [use.user for use in node.output().uses()]
            user_targets = [
                get_op_overload(user_node)
                for user_node in user_nodes
                if user_node.schema() != "(no schema)"
            ]
            has_mutable_target = any(
                target._schema.is_mutable for target in user_targets
            )

            if has_mutable_target:
                assert len(args) >= 4
                new_args = list(args)
                new_args[3] = True  # copy, override to True
                fx_node = self.fx_graph.call_function(
                    torch.ops.aten.to.dtype, tuple(new_args)
                )
                # temp hack to work around the issue https://github.com/pytorch/pytorch/issues/131679
                # When this issue is fixed, the clone node would be no longer needed
                clone_node = self.fx_graph.call_function(
                    torch.ops.aten.clone.default, (fx_node,)
                )
                output_name = node.output().debugName()
                self.name_to_node[output_name] = clone_node
                return

        self.convert_call_function_op(node)

    def convert_aten_add(self, node: torch._C.Node):
        if node.schema() == "(no schema)":
            if isinstance(node.inputsAt(0).type(), torch.ListType) and isinstance(
                node.inputsAt(1).type(), torch.ListType
            ):
                target = torch.ops.aten.add.t
            else:
                raise RuntimeError(f"unable to determined the target for {node}")
        else:
            target = get_op_overload(node)

        if target is torch.ops.aten.add.t:
            # special handle python list/tuple add: "aten::add.t(t[] a, t[] b) -> t[]" for
            # RuntimeError: aten::add() Expected a value of type 'List[t]' for argument 'a' but instead found type 'immutable_list'.
            args, _kwargs = self.get_args_kwargs(node, target._schema)
            output_name = node.output().debugName()
            self.name_to_node[output_name] = self.fx_graph.call_function(list_add, args)
        else:
            self.convert_call_function_op(node)

    def _check_prim_loop_support(self, node):
        inputs = list(node.inputs())

        # TODO: (1/N) stage.
        if inputs[0].debugName() not in self.name_to_constant:
            raise RuntimeError(
                "prim::Loop currently cannot run with dynamic value of number of iterations."
            )

        # Make sure the condition is not updated in the subblock.
        subblock = next(node.blocks())
        condition_output_name = next(subblock.outputs()).debugName()
        for node in subblock.nodes():
            if (
                node.outputsSize() == 1
                and node.output().debugName() == condition_output_name
            ):
                raise RuntimeError(
                    "prim::Loop currently cannot run with dynamic value of condition."
                )
            if node.outputsSize() >= 2:
                for outp in node.outputs():
                    if outp.debugName() == condition_output_name:
                        raise RuntimeError(
                            "prim::Loop currently cannot run with dynamic value of condition."
                        )

    def convert_prim_Loop(self, node: torch._C.Node):
        inputs = list(node.inputs())
        self._check_prim_loop_support(node)

        num_iterations = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        loop_local_arguments = [inp.debugName() for inp in inputs[2:]]

        global_arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            global_arguments = global_arguments.union(
                self.blocks_to_lifted_attrs[block]
            )

        global_arguments = list(global_arguments)

        subgraph_nodes, subgraph_converters = self._convert_block_to_subgraph(
            node, global_arguments
        )

        assert len(subgraph_nodes) == 1
        subgraph_converter = subgraph_converters[0]
        if not self.is_top_level_graph():
            self.name_update_from_subblock_to_parent = (
                self.name_update_from_subblock_to_parent.union(
                    subgraph_converter.name_update_from_subblock_to_parent
                )
            )

        fx_block_args = [
            self.get_fx_value_by_fqn(name)
            for name in loop_local_arguments + global_arguments
        ]
        for iter_idx in range(num_iterations):
            loop_node = self.fx_graph.call_function(
                execute_subgraph_from_prim_loop,
                # Check execute_node function for the expected arguments order.
                (
                    subgraph_nodes[0],
                    iter_idx,
                    len(loop_local_arguments),
                    *fx_block_args,
                ),
                {},
            )

            # Update the value of loop local variables.
            if node.outputsSize() >= 1:
                for i, outp in enumerate(node.outputs()):
                    output_name = outp.debugName()
                    self.name_to_node[output_name] = self.fx_graph.call_function(
                        operator.getitem,
                        (
                            loop_node,
                            i + 1,
                        ),  # + 1 because the 0th element is the condition.
                    )
                    fx_block_args[i] = self.name_to_node[output_name]

            # Update the value of global variables, whose values are modified inplace.

            for i, name in enumerate(
                subgraph_converter.name_update_from_subblock_to_parent
            ):
                self.name_to_node[name] = self.fx_graph.call_function(
                    operator.getitem,
                    (
                        loop_node,
                        i + node.outputsSize() + 1,
                    ),  # + 1 because the 0th element is the condition.
                )
                global_argument_index = global_arguments.index(name)
                fx_block_args[i + node.outputsSize() + global_argument_index] = (
                    self.name_to_node[name]
                )

    def _check_set_attr_in_if_block(self, if_node: torch._C.Node):
        for block in if_node.blocks():
            for node in block.nodes():
                if node.kind() == "prim::SetAttr":
                    raise RuntimeError(
                        "During converting prim::If to torch.cond, found prim::SetAttr op"
                        " which is not supported yet. Please file an issue if you come "
                        "across this error."
                    )

    def convert_prim_If(self, node: torch._C.Node):
        self._check_set_attr_in_if_block(node)

        inputs = list(node.inputs())
        assert len(inputs) == 1
        predicate = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)
        subgraph_nodes, _ = self._convert_block_to_subgraph(node, arguments)

        assert len(subgraph_nodes) == 2

        fx_block_args = [self.get_fx_value_by_fqn(name) for name in arguments]

        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        # prim::If can also have zero output.
        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = cond_node
        elif node.outputsSize() > 1:
            for i, output in enumerate(node.outputs()):
                output_name = output.debugName()
                getitem = self.fx_graph.call_function(operator.getitem, (cond_node, i))
                self.name_to_node[output_name] = getitem

    def convert_aten_Bool(self, node: torch._C.Node):
        self._convert_as_noop(node)

    def convert_prim_Enter(self, node: torch._C.Node):
        # export generally treats prim::Enter as noop
        # The only context manager export supports is aten::enable_grad.
        # Unfortunately, TorchScript does not support aten::enable_grad yet.
        # TODO: support aten::enable_grad in both TorchScript and Converter.
        return

    def convert_prim_Exit(self, node: torch._C.Node):
        # export treats prim::Exit as noop
        return

    def _convert_as_noop(self, node: torch._C.Node):
        # Converts the node as a no-op by mapping its output node as arg[0]

        target = get_op_overload(node)
        schema = target._schema

        args, _kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # _record_function_exit has side effect so we keep it in fx.graph
        # currently, _record_function_enter_new and _record_function_exit are
        # discarded during `retrace_as_exported_program`.
        target = torch.ops.profiler._record_function_exit
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        self.fx_graph.call_function(target, args)

    def convert_prim_tolist(self, node: torch._C.Node):
        # prim::tolist cannot be supported by `_convert_standard_operators`
        # since it requires call_method instead of call_function.
        target = "tolist"
        args = (self.get_fx_value_by_ir_value(next(node.inputs())),)
        fx_node = self.fx_graph.call_method(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_Uninitialized(self, node: torch._C.Node):
        # `prim::Uninitialized` is inserted by the compiler when it can prove
        # the value will never be used. It can be introduced by exceptions,
        # breaks, continues, and returns.
        # So we add a dummy constant to the graph.
        output_name = node.output().debugName()
        self.name_to_constant[output_name] = torch.Tensor()

    def _convert_standard_operators(self, node: torch._C.Node):
        target = kind_to_standard_operators[node.kind()]
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        fx_node = self.fx_graph.call_function(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_node(self, node: torch._C.Node):
        node_kind = node.kind()

        # Get handler based on namespace and operator name.
        # Provide a default node handler as well in case we don't find
        # matching converter for that.
        handler_func_name = ir_name_to_func_name(node_kind)
        handler_func = getattr(self, handler_func_name, self.convert_call_function_op)

        # str calls print function implemented in CPP. To avoid repeating
        # the entire logic here, we simply keep first line from node string (getting rid
        # of sub-blocks IR prints).
        node_str = "".join(str(node).split("\n")[:1])
        log.debug("[%s] converts [%s]", handler_func.__name__, node_str)
        try:
            handler_func(node)
        except Exception as e:
            raise RuntimeError(f"TS2EPConverter failed for node {node_kind}") from e

    def convert_graph_outputs(self):
        args = []
        outp_name_list = [outp.debugName() for outp in self.ts_graph.outputs()] + list(
            self.name_update_from_subblock_to_parent
        )
        for output_name in outp_name_list:
            if output_name in self.name_to_node:
                fx_node = self.name_to_node[output_name]
                # TODO: Revisit this later after HigherOrderOp design changes.
                # Currently, we cannot directly return input as output.
                if (
                    not self.is_top_level_graph()
                    and isinstance(fx_node, torch.fx.Node)
                    and fx_node.op == "placeholder"
                ):
                    fx_node = self.fx_graph.call_function(torch.clone, (fx_node,))
                args.append(fx_node)
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output_name),
                        target=output_name,
                    )
                )
            elif output_name in self.name_to_constant:
                args.append(self.name_to_constant[output_name])
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=ConstantArgument(
                            name=output_name, value=self.name_to_constant[output_name]
                        ),
                        target=output_name,
                    )
                )
            else:
                raise ValueError(f"Output {output_name} not found")

        if len(args) == 0:
            # Sub-block of prim::If can have zero output.
            self.fx_graph.output([])
        elif len(args) == 1:
            self.fx_graph.output(
                args[0]
            )  # Get rid of an extra list wrapped around final output.
        elif len(args) > 1:
            self.fx_graph.output(
                args
            )  # For prim::Loop and prim::If with multiple outputs.
        else:
            # Sub-block of prim::Loop can have multiple outputs.
            self.fx_graph.output(args)


class ExplainTS2FXGraphConverter(TS2FXGraphConverter):
    """
    Run TS2FXGraphConverter in an explain mode. It collects all failed operators conversions
    and provide that information to users. In order to collect all failed conversions, it
    also mocks some internal attributes (e.g., name_to_node).
    """

    class _DictMock(dict):
        def __init__(self, dict_data, mock_value):
            super().__init__(dict_data)
            self.mock_value = mock_value

        def __getitem__(self, key):
            # If the original dictionary has the key, return its value.
            # Otherwise, return the mock value.
            if not super().__contains__(key):
                return self.mock_value
            return super().__getitem__(key)

        def __contains__(self, key):
            return True

    def __init__(
        self,
        ts_graph: Union[torch._C.Graph, torch._C.Block],
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        super().__init__(
            ts_graph,
            name_to_param,
            name_to_buffer,
            blocks_to_lifted_attrs,
            name_to_non_tensor_attribute,
            name_to_constant,
            name_to_attribute_fqn,
        )

        # Data to keep track of unsupported nodes.
        self.unsupported_node_list: list[torch._C.Node] = []

        # Add mock to needed attributes.
        self.name_to_node = ExplainTS2FXGraphConverter._DictMock(
            self.name_to_node,
            # Dummy node.
            torch.fx.Node(
                None,  # type: ignore[arg-type]
                "mock",
                "call_function",
                lambda: None,
                (),
                {},
            ),
        )

    def explain(self):
        self.convert_graph_inputs()
        for node in self.ts_graph.nodes():
            self.convert_node(node)
        self.convert_graph_outputs()

    def convert_node(self, node):
        try:
            super().convert_node(node)
        except Exception:
            self.unsupported_node_list.append(node)


@contextmanager
def disable_logging(log):
    disabled = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = disabled


class TS2EPConverter:
    # TorchScript model to ExportedProgram converter
    def __init__(
        self,
        ts_model: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction],
        sample_args: tuple[Any, ...],
        sample_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.name_to_param: dict[str, torch.Tensor] = {}
        self.name_to_buffer: dict[str, torch.Tensor] = {}
        param_list = (
            list(self.ts_model.parameters())
            if not isinstance(self.ts_model, torch._C.ScriptFunction)
            else []
        )
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                # Check if tensor belongs to any parameter.
                if any(
                    (tensor == param).all()
                    for param in param_list
                    if tensor.shape == param.shape
                ):
                    self.name_to_param[k] = tensor
                else:
                    self.name_to_buffer[k] = tensor

        self.name_to_non_tensor_attributes: dict[str, Any] = {}
        self.name_to_constant: dict[str, Any] = {}

        self.lift_get_attr()

    def convert(self) -> ExportedProgram:
        log.info(
            """
TS2EPConverter logging starts from here.

INFO: (TORCH_LOGS="export" <cmd>)
    * Log TorchScript IR.

DEBUG: (TORCH_LOGS="+export" <cmd>), additionally
    * Log conversion IR by IR in a format of [<conversion handler name>] converts [<IR>].
        """
        )
        log.info("TorchScript graph\n\n%s\n", self.ts_graph)

        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        gm = graph_converter.convert()

        # Post-processing step to deal with quantized operators.
        replace_quantized_ops_with_standard_ops(gm)
        log.info("GraphModule: %s", gm.print_readable(print_output=False))

        ep = self.retrace_as_exported_program(
            gm,
            graph_converter.name_to_constant,
        )
        log.info("%s", ep)

        # Post-processing step to ensure ExportedProgram has the same state_dict as
        # the original TorchScript model. Throw warnings for additionally populated
        # state_dict entries.
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                if k not in ep.state_dict:
                    warnings.warn(
                        f"Manually populate {k} into state_dict ExportedProgram, but it is never used by the ExportedProgram.",
                        stacklevel=2,
                    )
                    ep.state_dict[k] = tensor

        return ep

    @disable_logging(log)
    def explain(self, print_output=True):
        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = ExplainTS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        graph_converter.explain()
        if len(graph_converter.unsupported_node_list) > 0:
            explain_str = "Unsupported nodes are found in the following list:"
            for i, n in enumerate(graph_converter.unsupported_node_list):
                node_str = "".join(str(n).split("\n")[:1])
                explain_str += f"\n\n    {i}. {n.kind()} [{node_str}]"
        else:
            explain_str = "Success!"
        if print_output:
            print(explain_str)
        return explain_str

    def retrace_as_exported_program(
        self,
        gm: torch.fx.GraphModule,
        name_to_constant: dict[str, Any],
    ):
        dynamic_shapes = _tree_map_with_path(
            lambda path, x: (
                [Dim.AUTO] * x.dim() if isinstance(x, torch.Tensor) else None
            ),
            self.sample_args,
        )

        # TODO: adjust input orders to match GraphSignature convention
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            pre_dispatch=True,
        )

        # Post-processing to make sure the ExportedProgram states are correct.
        # Because during conversion, we set tensor constants as GetAttr,
        # retracing cannot recognize them as tensor constants but instead
        # treat them as buffers. We need to set them again here.
        ep._constants.update(
            {
                k: v
                for k, v in name_to_constant.items()
                if isinstance(v, (torch.Tensor, torch.ScriptObject))
            }
        )
        for k in name_to_constant:
            ep.state_dict.pop(k, None)

        for spec in ep.graph_signature.input_specs:
            # Mark as constant tensors for erroneously traced buffers.
            if spec.kind == InputKind.BUFFER and spec.target in name_to_constant:
                assert isinstance(name_to_constant[spec.target], torch.Tensor), (
                    f"{type(name_to_constant[spec.target])} has been erroneously marked as buffer"
                )
                spec.kind = InputKind.CONSTANT_TENSOR
                spec.persistent = None
        ep.verifier().check(ep)

        return ep

    def lift_get_attr(self):
        # This function lifts multiple data types.

        #     1. Tensor constants attributes (e.g., self.data = torch.tensor([2,3]))
        #     to buffers. Currently, when there are tensor constants, export
        #     would error and ask users to register tensor constants as buffers.
        #     Since it is hard to manually do so for TorchScript models
        #     (e.g., source code is missing), this function automatically
        #     lifts tensor constants to be buffers.

        #     2. ScriptObbject to constant. It will then be converted to getattr in
        #     in the fx graph.
        #
        # This function should happen in TS2EPConverter instead of
        # TS2FXGraphConverter since it gets attributes from self.ts_model
        # which is not accessible in TS2FXGraphConverter. It is similar to where
        # we collect self.name_to_param and self.name_to_buffer.
        name_to_attribute_fqn: dict[str, str] = {}

        def get_attr(fqn: str):
            name = fqn.split(".")
            v = self.ts_model
            for n in name:
                v = getattr(v, n)
            return v

        def get_fqn(node: torch._C.Node):
            attr_name = node.s("name")
            input_name = node.input().debugName()
            root_attr_name = name_to_attribute_fqn[input_name]
            attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
            return attr_fqn

        def _dfs_get_attr(block):
            for node in block.nodes():
                if node.kind() == "prim::CreateObject":
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = ""

                if node.kind() == "prim::GetAttr":
                    attr_fqn = get_fqn(node)
                    value = get_attr(attr_fqn)
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = attr_fqn
                    if isinstance(value, torch.Tensor):
                        if attr_fqn not in self.name_to_buffer:
                            # Lift tensor constants to be a buffer
                            self.name_to_buffer[attr_fqn] = value
                    elif isinstance(value, torch.ScriptObject):
                        if attr_fqn not in self.name_to_constant:
                            self.name_to_constant[attr_fqn] = value
                    else:
                        self.name_to_non_tensor_attributes[attr_fqn] = value

                for subblock in node.blocks():
                    _dfs_get_attr(subblock)

        _dfs_get_attr(self.ts_graph)
