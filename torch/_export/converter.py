# mypy: allow-untyped-defs
import operator

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.export._trace

from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx import subgraph_rewriter
from torch.onnx.utils import _create_jit_graph

from torchgen.model import FunctionSchema


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

    replaced_patterns = subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def normalize_name(name: str) -> str:
    return name.replace(".", "_")


def ir_name_to_func_name(name: str) -> str:
    """prim::If -> convert_prim_If"""
    name_list = name.split("::")
    return "convert_" + "_".join(name_list)


def get_node_for_param_and_buffer(fx_graph, name, is_top_level_graph):
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
    torch.bfloat16: 15,
}


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
kind_to_standard_operators = {
    "prim::TupleIndex": operator.getitem,
    "aten::__is__": operator.is_,
    "aten::__isnot__": operator.is_not,
    "aten::__not__": operator.not_,
    "aten::__contains__": operator.contains,
    "prim::dtype": get_dtype_as_int,
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


def get_block_to_lifted_attrs(graph: torch._C.Graph) -> Dict[torch._C.Block, Set[str]]:
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
        A mapping of blocks to a set of FQNs of its lifted attributes.
    """

    # A map from a block to its expected to be lifted arguments.
    blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]] = dict()

    # Reference map stores the input (i.e., src) and output (i.e., dest) IR of a
    # GetAttr node. By traversing this reference map, we can figure out the
    # full IR aliasing pass and figure out the FQN of an attribute.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_parent_map["%2"] = "%1"
    node_to_parent_map: Dict[str, str] = dict()

    # Used for reconstructing the FQN of an attribute based on the reference map.
    # In nutshell, for each GetAttr call, GetAttr(input IR, attribute name) -> output IR
    # This name map stores which attribute name is called for a src IR --> dest IR action.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_attr_name["%2"] = "linear"
    node_to_attr_name: Dict[str, str] = dict()

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
        arguments: Set[str] = set()
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

    return blocks_to_lifted_attrs


def get_op_overload(node: torch._C.Node):
    schema_str = node.schema()
    schema = FunctionSchema.parse(schema_str)
    ns, op_name = str(schema.name.name).split("::")
    override = schema.name.overload_name

    try:
        op_overload_mod = getattr(torch.ops, ns)
        op_overload_packet = getattr(op_overload_mod, op_name)
        if override:
            op_overload = getattr(op_overload_packet, override)
        else:
            op_overload = op_overload_packet.default
    except Exception as e:
        raise RuntimeError(
            f"Unable to find operator {node.kind()} with schema {node.schema}"
        ) from e

    return op_overload


class TS2FXGraphConverter:
    def __init__(
        self,
        ts_graph: Union[torch._C.Graph, torch._C.Block],
        name_to_param_map: Dict[str, torch.Tensor],
        name_to_buffer_map: Dict[str, torch.Tensor],
        blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]],
    ):
        self.ts_graph = ts_graph
        self.name_to_param_map = name_to_param_map
        self.name_to_buffer_map = name_to_buffer_map

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: List[InputSpec] = []
        self.output_specs: List[OutputSpec] = []

        self.name_to_node: Dict[
            str, Union[torch.fx.Node, List[torch.fx.Node], Dict[Any, torch.fx.Node]]
        ] = {}
        self.constant_map: Dict[str, Any] = {}
        self.attribute_map: Dict[str, Any] = {}
        self.tensor_constants: Dict[str, torch.Tensor] = {}

        self.subgraphs: Dict[str, torch.fx.GraphModule] = {}

        self.blocks_to_lifted_attrs = blocks_to_lifted_attrs

        # Populate methods for the standard operators.
        for k in kind_to_standard_operators.keys():
            handler_func_name = ir_name_to_func_name(k)
            # Create an indirect function call:
            # convert_<namespace>_<opname> --> lambda node: _convert_standard_operator(node)
            setattr(
                self,
                handler_func_name,
                lambda node: self._convert_standard_operators(node),
            )

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
                kwargs[schema_arg.name] = self.get_fx_value(input)
            else:
                args.append(self.get_fx_value(input))

        return tuple(args), kwargs

    def get_fx_value(self, value: torch._C.Value):
        value_name = value.debugName()
        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        elif value_name in self.attribute_map:
            attr_name = self.attribute_map[value_name]
            if attr_name in self.name_to_node:
                input_node = self.name_to_node[attr_name]
                return input_node
            else:
                raise ValueError(f"Value {attr_name} not found")
        elif value_name in self.constant_map:
            return self.constant_map[value_name]
        else:
            raise ValueError(f"Input {value_name} not found")

    def convert(self) -> torch.fx.GraphModule:
        self.convert_graph_inputs()

        for node in self.ts_graph.nodes():
            self.convert_node(node)

        self.convert_graph_outputs()

        # Pass parameter and buffer to the root for lookup.
        gm = torch.fx.GraphModule(
            {**self.subgraphs, **self.name_to_param_map, **self.name_to_buffer_map},
            self.fx_graph,
        )

        inplace_optimize_sym_size_div(gm)

        gm.graph.lint()

        return gm

    def convert_graph_inputs(self):
        for graph_input in self.ts_graph.inputs():
            name = graph_input.debugName()
            normalized_name = normalize_name(name)

            if name in self.name_to_param_map:
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = get_node_for_param_and_buffer(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_buffer_map:
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
                fx_node = get_node_for_param_and_buffer(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            else:
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = self.fx_graph.placeholder(normalized_name)

            self.name_to_node[name] = fx_node

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
                # lift tensor constant as a placeholder
                placeholder_name = f"constant_{name}"
                fx_node = self.fx_graph.placeholder(placeholder_name)
                self.name_to_node[name] = fx_node
                self.tensor_constants[placeholder_name] = node.t("value")

                self.input_specs.append(
                    InputSpec(
                        InputKind.CONSTANT_TENSOR,
                        arg=TensorArgument(name=placeholder_name),
                        target=placeholder_name,
                    )
                )

                value = fx_node
            elif constant_kind == "ival":
                value = node.ival("value")
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")
        else:
            value = None

        self.constant_map[name] = value

    def convert_prim_device(self, node: torch._C.Node):
        input_type = node.input().type()
        if input_type.isSubtypeOf(torch._C.TensorType.get()):
            device = input_type.device()  # type: ignore[attr-defined]
            output_name = node.output().debugName()
            self.constant_map[output_name] = device
        else:
            raise ValueError(f"Unsupported JitType ({input_type}) when get device")

    def convert_prim_GetAttr(self, node: torch._C.Node):
        def get_attr(name: str):
            if name in self.attribute_map:
                return self.attribute_map[name]
            else:
                raise ValueError(f"Attribute {name} not found")

        output_name = node.output().debugName()

        attr_name = node.s("name")
        input_name = node.input().debugName()

        root_attr_name = get_attr(input_name)
        self.attribute_map[output_name] = (
            f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
        )

    def convert_call_function_op(self, node: torch._C.Node):
        target = get_op_overload(node)

        if target is torch.ops.aten.size.int:
            target = torch.ops.aten.sym_size.int

        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: covnert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_TupleConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        self._convert_prim_iterator(node)

    def _convert_prim_iterator(self, node: torch._C.Node):
        output_list = []
        for inp in node.inputs():
            output_list.append(self.get_fx_value(inp))

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_list

    def convert_prim_DictConstruct(self, node: torch._C.Node):
        output_dict = {}
        k, v = None, None
        for i, inp in enumerate(node.inputs()):
            # We assume key value are stored in pair in the DictConstruct.
            # The first element is the key and the following is the value.
            if i % 2 == 0:
                k = self.get_fx_value(inp)
            else:
                v = self.get_fx_value(inp)
                assert (
                    k is not None and v is not None
                ), "DictConstruct has an empty key value pair."
                output_dict[k] = v
                k, v = None, None

        assert (
            k is None and v is None
        ), "DictConstruct has an odd number of elements (violating our assumption)."

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
            inp = self.get_fx_value(node.input())
            fx_node = self.fx_graph.call_function(operator.getitem, (inp, i))
            self.name_to_node[outp_name] = fx_node

    def convert_aten_Int(self, node: torch._C.Node):
        # converts aten::Int as aten._to_copy + aten::_local_scalar_dense
        target = torch.ops.aten._to_copy.default
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: covnert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # converts prim::NumToTensor as aten.scalar_tensor
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value(input) for input in node.inputs())

        fx_node = self.fx_graph.call_function(target, args)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        output_name = node.output().debugName()
        self.attribute_map[output_name] = ""

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
            if arg1_name in self.tensor_constants:
                tensor_constant = self.tensor_constants[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.tensor_constants[arg1_name].item()

                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: covnert sourceRange() into stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    output_name = node.output().debugName()
                    self.name_to_node[output_name] = fx_node
                    return

        self.convert_call_function_op(node)

    def convert_aten___getitem__(self, node: torch._C.Node):
        input_container, index = tuple(
            self.get_fx_value(input) for input in node.inputs()
        )
        fx_node = self.fx_graph.call_function(
            operator.getitem, (input_container, index)
        )
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_If(self, node: torch._C.Node):
        inputs = list(node.inputs())
        assert len(inputs) == 1
        predicate = self.get_fx_value(inputs[0])

        # Get union of inputs to blocks
        arguments = set()
        for block in node.blocks():
            block_args = set()

            # TODO: block.inputs(), not sure what theyre used for

            for block_node in block.nodes():
                for block_node_in in block_node.inputs():
                    if block_node_in.debugName() in self.name_to_node:
                        block_args.add(block_node_in.debugName())

            arguments.update(block_args)

        # Lift parameters as inputs.
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)

        # Convert blocks to subgraphs
        subgraph_nodes = []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(
                block, dict(), dict(), self.blocks_to_lifted_attrs
            )
            subgraph_converter.constant_map = self.constant_map
            subgraph_converter.attribute_map = self.attribute_map

            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))

        assert len(subgraph_nodes) == 2

        fx_block_args = [self.name_to_node[arg_name] for arg_name in arguments]
        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        output_name = node.output().debugName()
        self.name_to_node[output_name] = cond_node

    def convert_aten_Bool(self, node: torch._C.Node):
        self._convert_as_noop(node)

    def _convert_as_noop(self, node: torch._C.Node):
        # Converts the node as a no-op by mapping its output node as arg[0]

        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_enter_new(self, node: torch._C.Node):
        target = torch.ops.profiler._record_function_enter_new
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        fx_node = self.fx_graph.call_function(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # _record_function_exit has side effect so we keep it in fx.graph
        # currently, _record_function_enter_new and _record_function_exit are
        # discarded during `retrace_as_exported_program`.
        target = torch.ops.profiler._record_function_exit
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        self.fx_graph.call_function(target, args)

    def _convert_standard_operators(self, node: torch._C.Node):
        target = kind_to_standard_operators[node.kind()]
        args = tuple(self.get_fx_value(input) for input in node.inputs())
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
        handler_func(node)

    def convert_graph_outputs(self):
        args = []
        for graph_output in self.ts_graph.outputs():
            output_name = graph_output.debugName()
            if output_name in self.name_to_node:
                args.append(self.name_to_node[output_name])
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output_name),
                        target=output_name,
                    )
                )
            elif output_name in self.constant_map:
                args.append(self.constant_map[output_name])
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=ConstantArgument(
                            name=output_name, value=self.constant_map[output_name]
                        ),
                        target=output_name,
                    )
                )
            else:
                raise ValueError(f"Output {output_name} not found")

        self.fx_graph.output(
            args[0]
        )  # Get rid of an extra list wrapped around final output.


class TS2EPConverter:
    # TorchScript model to ExportedProgram converter
    def __init__(
        self,
        ts_model: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction],
        sample_args: Tuple[Any, ...],
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.name_to_param_map: Dict[str, torch.Tensor] = (
            dict(ts_model.named_parameters())
            if isinstance(ts_model, torch.jit.ScriptModule)
            else dict()
        )
        self.name_to_buffer_map: Dict[str, torch.Tensor] = (
            dict(ts_model.named_buffers())
            if isinstance(ts_model, torch.jit.ScriptModule)
            else dict()
        )

    def convert(self) -> ExportedProgram:
        blocks_to_lifted_attrs = get_block_to_lifted_attrs(self.ts_graph)

        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param_map,
            self.name_to_buffer_map,
            blocks_to_lifted_attrs,
        )
        gm = graph_converter.convert()
        ep = self.retrace_as_exported_program(gm, graph_converter.tensor_constants)
        return ep

    def retrace_as_exported_program(self, gm: torch.fx.GraphModule, tensor_constants):
        # TODO: adjust input orders to match GraphSignature convention
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            strict=False,
            pre_dispatch=True,
        )
        return ep
