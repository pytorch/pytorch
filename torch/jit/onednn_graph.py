import numbers
from math import prod
from typing import List

import torch
import torch._C._onednn as llga
from torch.fx import Node


class OnednnGraph:
    def __init__(self):
        self.graph = llga.graph(llga.engine.cpu)
        self.engine = llga.engine(llga.engine.cpu, 0)
        self.stream = llga.stream(self.engine)
        self.desc_id_to_node_map: dict[int, Node] = {}
        self.desc_to_scalar_data: dict[int, numbers.Number] = {}
        self.dtype_map: dict[torch.dtype, llga.logical_tensor] = {
            torch.float16: llga.logical_tensor.f16,
            torch.bfloat16: llga.logical_tensor.bf16,
            torch.float32: llga.logical_tensor.f32,
            torch.int32: llga.logical_tensor.s32,
            torch.int8: llga.logical_tensor.s8,
            torch.uint8: llga.logical_tensor.u8,
            torch.bool: llga.logical_tensor.boolean,
            torch.int64: llga.logical_tensor.dt_undef,
            torch.float64: llga.logical_tensor.dt_undef,
            torch.complex64: llga.logical_tensor.dt_undef,
        }
        self.reverse_dtype_map: dict[llga.logical_tensor, torch.dtype] = {
            self.dtype_map[key]: key for key in self.dtype_map
        }
        self.current_id = 0
        self.desc_id_to_queried_desc = {}
        self.desc_ids_with_any_layout = set()
        self.partitions = None
        self.is_inference = False

    def get_partitions(self, policy=llga.partition.fusion):
        if self.partitions:
            return self.partitions
        self.partitions = self.graph.get_partitions(policy)
        self.set_any_layout()
        return self.partitions

    # based on
    # https://github.com/oneapi-src/oneDNN/blob/6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e/tests/benchdnn/graph/helpers_any_layout.hpp#L29-L96
    # Set tensor layout to "any" when used only by LLGA partitions
    def set_any_layout(self, partitions=None):
        if partitions is None:
            partitions = self.partitions
        self.desc_ids_with_any_layout = set()
        # map from output id to all supported flags of supported partitions
        output_to_flag_map = {}

        # Initialize map of supported tensors
        for p in self.partitions:
            p_is_supported = p in partitions and p.is_supported()
            for out_desc in p.get_out_ports():
                id = out_desc.get_id()
                if p_is_supported and id not in output_to_flag_map:
                    output_to_flag_map[id] = []
            for in_desc in p.get_in_ports():
                id = in_desc.get_id()
                if id in output_to_flag_map:
                    output_to_flag_map[id].append(p_is_supported)
        for p in partitions:
            if not p.is_supported():
                continue
            for in_desc in p.get_in_ports():
                id = in_desc.get_id()
                if id not in output_to_flag_map:
                    continue
                flags = output_to_flag_map[id]
                # if all uses of in_desc are supported use "any"
                if all(flags):
                    self.desc_ids_with_any_layout.add(id)

    def update_input_descs(
        self,
        descs: List[llga.logical_tensor],
        aten_inputs: List[torch.Tensor],
        cache_weight: bool = False,
    ):
        assert len(descs) == len(aten_inputs)
        for i, (d, at_in) in enumerate(zip(descs, aten_inputs)):
            make_constant = cache_weight and isinstance(at_in, torch.nn.Parameter)
            property_type = (
                llga.logical_tensor.property_type.constant
                if make_constant
                else llga.logical_tensor.property_type.variable
            )
            descs[i] = llga.logical_tensor(
                d.get_id(),
                d.get_data_type(),
                at_in.size(),
                at_in.stride(),
                property_type,
            )
        return descs

    def get_compiled_output_descs(
        self, cp: llga.compiled_partition, out_descs: List[llga.logical_tensor]
    ):
        return [cp.query_logical_tensor(desc.get_id()) for desc in out_descs]

    def compile_partition(self, p: llga.partition, inputs: List[llga.logical_tensor]):
        outputs = p.get_out_ports()
        for i, input in enumerate(inputs):
            id = input.get_id()
            if id in self.desc_id_to_queried_desc:
                inputs[i] = self.desc_id_to_queried_desc[id]
        for i, output in enumerate(outputs):
            id = output.get_id()
            if id in self.desc_ids_with_any_layout:
                outputs[i] = llga.logical_tensor(
                    id,
                    output.get_data_type(),
                    output.get_dims(),
                    llga.logical_tensor.layout_type.any,
                    output.get_property_type(),
                )
        cpart = p.compile(inputs, outputs, self.engine)
        for i, output in enumerate(outputs):
            id = output.get_id()
            outputs[i] = cpart.query_logical_tensor(id)
            self.desc_id_to_queried_desc[id] = outputs[i]
        return cpart

    def add_op(self, op_kind, name, inputs, outputs=None, kwargs=None):
        if outputs is None:
            outputs = []
        if outputs:
            id = outputs[0].get_id()
        else:
            id = self.generate_id()
        op = llga.op(id, op_kind, name)
        if kwargs is None:
            kwargs = {}
        for attr_key in kwargs:
            if isinstance(attr_key, str):
                if hasattr(llga.op, attr_key):
                    op.set_attributes(getattr(llga.op, attr_key), kwargs[attr_key])
            else:
                op.set_attributes(attr_key, kwargs[attr_key])
        op.add_inputs(inputs)
        if outputs:
            op.add_outputs(outputs)
        self.graph.add_op(op, True)
        return id

    def generate_id(self):
        id = self.current_id
        self.current_id += 1
        return id

    def create_onednn_descs_from_node(
        self, node: Node, ptype=llga.logical_tensor.property_type.variable
    ) -> List[llga.logical_tensor]:
        assert node.op in ["placeholder", "call_function"]
        if isinstance(node.meta["val"], (list, tuple)):
            outputs = [
                self.create_onednn_desc_from_meta(val, ptype)
                for val in node.meta["val"]
            ]
        else:
            outputs = [self.create_onednn_desc_from_meta(node.meta["val"], ptype)]
        for desc in outputs:
            self.register_node_by_desc(node, desc)
        return outputs

    def create_onednn_desc_from_meta(
        self, val, ptype=llga.logical_tensor.property_type.variable
    ) -> llga.logical_tensor:
        if isinstance(val, torch.SymInt):
            dtype = llga.logical_tensor.dt_undef
            size = [1]
            stride = [1]
        else:
            dtype = self.dtype_map[val.dtype]
            size = list(val.size())
            stride = list(val.stride())
        # TODO: workaround to reset stride due to oneDNN bug
        # for cases stride is larger than number of elements in tensor
        # and in decreasing order (contiguous)
        if (
            len(size)
            and size[0] == 1
            and stride[0] > prod(size)
            and stride == sorted(stride, reverse=True)
        ):
            stride[0] = prod(size)
        onednn_desc = llga.logical_tensor(
            self.generate_id(), dtype, size, stride, ptype
        )
        return onednn_desc

    def create_onednn_desc_from_scalar(self, scalar, dtype=None) -> llga.logical_tensor:
        fake_tensor = torch.tensor([scalar], dtype=dtype)
        desc = self.create_onednn_desc_from_meta(
            fake_tensor, ptype=llga.logical_tensor.property_type.constant
        )
        self.register_scalar_data(scalar, desc, dtype)
        return desc

    def overwrite_scalar_args(self, args, cast_scalar=True):
        args = list(args)
        if cast_scalar:
            assert len(args) == 2
            assert any(isinstance(arg, llga.logical_tensor) for arg in args)
            for arg in args:
                if isinstance(arg, llga.logical_tensor):
                    torch_type = self.reverse_dtype_map[arg.get_data_type()]
                    break
        for arg_idx, arg in enumerate(args):
            # We only handle scalars, not lists of constant scalars
            if isinstance(arg, numbers.Number):
                # arg is a scalar value, we get a logical_tensor of shape=()
                cast_type = torch_type if cast_scalar else None
                args[arg_idx] = self.create_onednn_desc_from_scalar(arg, cast_type)
        return args

    def register_node_by_desc(self, node, desc):
        if isinstance(desc, llga.logical_tensor):
            desc = desc.get_id()
        self.desc_id_to_node_map[desc] = node

    def get_node_from_desc(self, desc):
        if isinstance(desc, llga.logical_tensor):
            desc = desc.get_id()
        return self.desc_id_to_node_map[desc]

    def register_scalar_data(self, data, desc, dtype=None):
        if isinstance(desc, llga.logical_tensor):
            desc = desc.get_id()
        self.desc_to_scalar_data[desc] = (data, dtype)

    def get_scalar_data_from_desc(self, desc):
        if isinstance(desc, llga.logical_tensor):
            desc = desc.get_id()
        return self.desc_to_scalar_data[desc]

    def get_args_to_onednn_partition_order(self, onednn_partition, node_args):
        onednn_input_names = [
            in_desc.get_id()
            if in_desc.get_id() in self.desc_to_scalar_data.keys()
            else self.get_node_from_desc(in_desc).name
            for in_desc in onednn_partition.get_in_ports()
        ]
        arg_names = [n.name for n in node_args]

        # Get the index in args if arg exists in args, otherwise get the relevant scalar data from graph
        args_to_onednn_order = []
        for name in onednn_input_names:
            if name in arg_names:
                args_to_onednn_order.append(arg_names.index(name))
            else:
                scalar, dtype = self.get_scalar_data_from_desc(name)
                scalar = torch.tensor([scalar], dtype=dtype)
                if hasattr(scalar, "constant"):
                    scalar = scalar.constant
                args_to_onednn_order.append(scalar)
        return args_to_onednn_order
