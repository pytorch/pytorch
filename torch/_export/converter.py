# mypy: allow-untyped-defs
import builtins
import logging
import operator
import warnings
from contextlib import contextmanager
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


log = logging.getLogger(__name__)


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
kind_to_standard_operators = {
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
    blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]] = {}

    # Reference map stores the input (i.e., src) and output (i.e., dest) IR of a
    # GetAttr node. By traversing this reference map, we can figure out the
    # full IR aliasing pass and figure out the FQN of an attribute.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_parent_map["%2"] = "%1"
    node_to_parent_map: Dict[str, str] = {}

    # Used for reconstructing the FQN of an attribute based on the reference map.
    # In nutshell, for each GetAttr call, GetAttr(input IR, attribute name) -> output IR
    # This name map stores which attribute name is called for a src IR --> dest IR action.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_attr_name["%2"] = "linear"
    node_to_attr_name: Dict[str, str] = {}

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


def get_attribute_fqn_from_ts_node(
    name_to_attribute_fqn: Dict[str, str], node: torch._C.Node
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
            f"Unable to find operator {node.kind()} with schema {node.schema}"
        ) from e

    return op_overload


class TS2FXGraphConverter:
    def __init__(
        self,
        ts_graph: Union[torch._C.Graph, torch._C.Block],
        name_to_param: Dict[str, torch.Tensor],
        name_to_buffer: Dict[str, torch.Tensor],
        blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]],
        name_to_non_tensor_attribute: Dict[str, Any],
    ):
        self.ts_graph = ts_graph
        self.name_to_param = name_to_param
        self.name_to_buffer = name_to_buffer

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: List[InputSpec] = []
        self.output_specs: List[OutputSpec] = []

        self.name_to_node: Dict[
            str, Union[torch.fx.Node, List[torch.fx.Node], Dict[Any, torch.fx.Node]]
        ] = {}
        self.constant_map: Dict[str, Any] = {}

        # Mapping from torchscript node output name to attribute fully qualified name
        self.name_to_attribute_fqn: Dict[str, str] = {}

        self.name_to_tensor_constants: Dict[str, torch.Tensor] = {}

        # Mapping from fully qualified name to real values or a fx graph node
        # During convert, this represents the current value of a non-tensor attribute
        # One use case is:
        #   def forward(self, x):
        #        c1 = self.count
        #        self.count += 1
        #        c2 = self.count
        #        return x + c1 + c2
        self.name_to_non_tensor_attribute_node: Dict[str, Any] = {}

        # Mapping from fully qualified name to initial real values inputs
        # We separate it from self.name_to_non_tensor_attribute_node since
        # we need initial real value input when we construct fx.GraphModule
        self.name_to_non_tensor_attribute: Dict[str, Any] = name_to_non_tensor_attribute

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
            {
                **self.subgraphs,
                **self.name_to_param,
                **self.name_to_buffer,
                **self.name_to_tensor_constants,
                **self.name_to_non_tensor_attribute,
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
                fx_node = get_node_for_param_and_buffer(
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
                fx_node = get_node_for_param_and_buffer(
                    self.fx_graph, name, self.is_top_level_graph()
                )
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
                self.name_to_tensor_constants[alias_name] = node.t("value")
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
        # Build fully qulified name
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = attr_fqn

        attr_value = node.output()
        if self.is_top_level_graph():
            if attr_value.type().annotation_str == "Tensor":
                # We insert a get_attr node due to two reasons.
                # First, ts graph does not lift tensor constants as input nodes. So
                # tensor constants may be ignored by in convert_graph_inputs().
                # Second, attr_fqn may have been written to via SetAttr. Two
                # GetAttr may give different values.
                self.name_to_node[output_name] = self.fx_graph.get_attr(attr_fqn)
            else:
                if attr_fqn not in self.name_to_non_tensor_attribute_node:
                    self.name_to_non_tensor_attribute_node[
                        attr_fqn
                    ] = self.name_to_non_tensor_attribute[attr_fqn]
                self.name_to_node[output_name] = self.name_to_non_tensor_attribute_node[
                    attr_fqn
                ]
        else:
            # Special support for if blocks which do not allow SetAttr TorchScript
            # node and get_attr FX Graph Node.
            if attr_value.type().annotation_str == "Tensor":
                self.name_to_node[output_name] = self.name_to_node[attr_fqn]

    def convert_prim_SetAttr(self, node: torch._C.Node):
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        attr_value = tuple(node.inputs())[1]
        ts_graph_tensor_input = self.get_fx_value(attr_value)
        if attr_value.type().annotation_str == "Tensor":
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

        # TODO: covnert sourceRange() into stack_trace
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
        # Converts prim::NumToTensor as aten.scalar_tensor.
        # prim::NumToTensor IRs are currently triggered by:
        # .size() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L950
        # .numel() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L971
        # For both of those APIs, torch.jit.trace implicitly sets the output tensor type
        # to be LongTensor.
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value(input) for input in node.inputs())

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
            if arg1_name in self.name_to_tensor_constants:
                tensor_constant = self.name_to_tensor_constants[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.name_to_tensor_constants[arg1_name].item()

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
        predicate = self.get_fx_value(inputs[0])

        def _identify_inputs_as_arguments(entry):
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
            arguments: Set[str] = set()
            for block in entry.blocks():
                for block_node in block.nodes():
                    for block_node_in in block_node.inputs():
                        if (
                            block_node_in.debugName() in self.name_to_node
                            and block_node_in.debugName()
                            not in self.name_to_attribute_fqn
                        ):
                            arguments.add(block_node_in.debugName())
                    arguments = arguments.union(
                        _identify_inputs_as_arguments(block_node)
                    )
            return arguments

        # Find inputs.
        arguments = _identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)

        # Convert blocks to subgraphs
        subgraph_nodes = []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(
                block, {}, {}, self.blocks_to_lifted_attrs, {}
            )
            subgraph_converter.constant_map = self.constant_map
            subgraph_converter.name_to_attribute_fqn = self.name_to_attribute_fqn

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

        fx_block_args = []
        for arg_name in arguments:
            if arg_name in self.name_to_node:
                arg_node = self.name_to_node[arg_name]
                fx_block_args.append(arg_node)
            elif arg_name in self.name_to_non_tensor_attribute_node:
                arg_node = self.name_to_non_tensor_attribute_node[arg_name]
                fx_block_args.append(arg_node)
            elif arg_name in self.name_to_non_tensor_attribute:
                arg_value = self.name_to_non_tensor_attribute[arg_name]
                fx_block_args.append(arg_value)
            else:
                raise ValueError(f"Attribute {arg_name} not found")

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

        args, kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # _record_function_exit has side effect so we keep it in fx.graph
        # currently, _record_function_enter_new and _record_function_exit are
        # discarded during `retrace_as_exported_program`.
        target = torch.ops.profiler._record_function_exit
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        self.fx_graph.call_function(target, args)

    def convert_prim_tolist(self, node: torch._C.Node):
        # prim::tolist cannot be supported by `_convert_standard_operators`
        # since it requires call_method instead of call_function.
        target = "tolist"
        args = (self.get_fx_value(next(node.inputs())),)
        fx_node = self.fx_graph.call_method(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_Uninitialized(self, node: torch._C.Node):
        # `prim::Uninitialized` is inserted by the compiler when it can prove
        # the value will never be used. It can be introduced by exceptions,
        # breaks, continues, and returns.
        # So we add a dummy constant to the graph.
        output_name = node.output().debugName()
        self.constant_map[output_name] = torch.Tensor()

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

        if len(args) == 1:
            self.fx_graph.output(
                args[0]
            )  # Get rid of an extra list wrapped around final output.
        else:
            # Sub-block of prim::If can have zero output.
            self.fx_graph.output([])


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
        name_to_param: Dict[str, torch.Tensor],
        name_to_buffer: Dict[str, torch.Tensor],
        blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]],
        name_to_non_tensor_attribute: Dict[str, Any],
    ):
        super().__init__(
            ts_graph,
            name_to_param,
            name_to_buffer,
            blocks_to_lifted_attrs,
            name_to_non_tensor_attribute,
        )

        # Data to keep track of unsupported nodes.
        self.unsupported_node_list: List[torch._C.Node] = []

        # Add mock to needed attributes.
        self.name_to_node = ExplainTS2FXGraphConverter._DictMock(
            self.name_to_node,
            # Dummy node.
            torch.fx.Node(
                None,
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
        sample_args: Tuple[Any, ...],
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.name_to_param: Dict[str, torch.Tensor] = {}
        self.name_to_buffer: Dict[str, torch.Tensor] = {}
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

        self.name_to_non_tensor_attributes: Dict[str, Any] = {}

        self.lift_tensor_constants_to_buffer()

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

        blocks_to_lifted_attrs = get_block_to_lifted_attrs(self.ts_graph)

        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
        )
        gm = graph_converter.convert()
        log.info("GraphModule: %s", gm.print_readable(print_output=False))

        ep = self.retrace_as_exported_program(
            gm, graph_converter.name_to_tensor_constants
        )
        log.info("%s", ep)

        # Post-processing step to ensure ExportedProgram has the same state_dict as
        # the original TorchScript model. Throw warnings for additionally populated
        # state_dict entries.
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                if k not in ep.state_dict:
                    warnings.warn(
                        f"Manually populate {k} into state_dict ExportedProgram, but it is never used by the ExportedProgram."
                    )
                    ep.state_dict[k] = tensor

        return ep

    @disable_logging(log)
    def explain(self, print_output=True):
        blocks_to_lifted_attrs = get_block_to_lifted_attrs(self.ts_graph)

        graph_converter = ExplainTS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
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
        self, gm: torch.fx.GraphModule, tensor_constants: Dict[str, torch.Tensor]
    ):
        # TODO: adjust input orders to match GraphSignature convention
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            strict=False,
            pre_dispatch=True,
        )

        # Post-processing to make sure the ExportedProgram states are correct.
        # Because during conversion, we set tensor constants as GetAttr,
        # retracing cannot recognize them as tensor constants but instead
        # treat them as buffers. We need to set them again here.
        ep._constants = {**ep._constants, **tensor_constants}
        for k in tensor_constants:
            ep.state_dict.pop(k, None)
        for spec in ep.graph_signature.input_specs:
            # Mark as constant tensors for erroneously traced buffers.
            if spec.kind == InputKind.BUFFER and spec.target in tensor_constants:
                spec.kind = InputKind.CONSTANT_TENSOR
        ep.verifier().check(ep)

        return ep

    def lift_tensor_constants_to_buffer(self):
        # This function lifts tensor constants attributes (e.g., self.data = torch.tensor([2,3]))
        # to buffers. Currently, when there are tensor constants, export
        # would error and ask users to register tensor constants as buffers.
        # Since it is hard to manually do so for TorchScript models
        # (e.g., source code is missing), this function automatically
        # lifts tensor constants to be buffers.
        # This function should happen in TS2EPConverter instead of
        # TS2FXGraphConverter since it gets attributes from self.ts_model
        # which is not accessable in TS2FXGraphConverter. It is similar to where
        # we collect self.name_to_param and self.name_to_buffer.
        name_to_attribute_fqn: Dict[str, str] = {}

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
                    else:
                        self.name_to_non_tensor_attributes[attr_fqn] = value

                for subblock in node.blocks():
                    _dfs_get_attr(subblock)

        _dfs_get_attr(self.ts_graph)
