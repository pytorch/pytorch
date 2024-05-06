import torch
import torch._C._onednn_graph as onednn_graph
from torch.profiler import record_function

onednn_graph_dtype_map: dict[torch.dtype, onednn_graph.logical_tensor] = {
    torch.float16: onednn_graph.logical_tensor.f16,
    torch.bfloat16: onednn_graph.logical_tensor.bf16,
    torch.float32: onednn_graph.logical_tensor.f32,
    torch.int32: onednn_graph.logical_tensor.s32,
    torch.int8: onednn_graph.logical_tensor.s8,
    torch.uint8: onednn_graph.logical_tensor.u8,
    torch.bool: onednn_graph.logical_tensor.boolean,
}

onednn_graph_reverse_dtype_map: dict[onednn_graph.logical_tensor, torch.dtype] = {
    onednn_graph_dtype_map[key]: key for key in onednn_graph_dtype_map
}


class OnednnGraphPartitionModule:
    def __init__(self, name, output_tensor_description):
        super().__init__()
        self.is_opaque = True
        self.__name__ = name
        self.partition = None
        self.compiled_partition = None
        self.kernel = None
        self.tags = []
        # cache output tensors
        self.output_tensors = []
        self.namespace = "onednn_graph"
        # list of tuples of shape, stride & datatype info
        self.output_tensor_description = output_tensor_description

        # oneDNN Graph specific data structures
        self.graph = onednn_graph.graph(onednn_graph.engine.cpu)
        self.engine = onednn_graph.engine(onednn_graph.engine.cpu, 0)
        self.stream = onednn_graph.stream(self.engine)
        self.lt_idx = 0
        self.op_idx = 0
        self.partitions = None
        self.is_inference = False
        self.input_logical_tensors = []
        self.output_logical_tensors = []

    def add_op(self, op):
        self.graph.add_op(op, True)

    def create_output_tensors(self):
        retVal = []
        for output_tensor in self.output_logical_tensors:
            retVal.append(
                torch.empty_strided(
                    output_tensor.get_dims(), output_tensor.get_strides()
                ).to(onednn_graph_reverse_dtype_map[output_tensor.get_data_type()])
            )
        return retVal

    def create_op(self, op_name, op_str):
        op = onednn_graph.op(self.op_idx, op_name, op_str)
        self.op_idx += 1
        return op

    def add_input_logical_tensor(self, input_tensor):
        self.input_logical_tensors.append(input_tensor)

    def add_output_logical_tensor(self, output_tensor):
        self.output_logical_tensors.append(output_tensor)

    def finalize_graph(self):
        self.graph.finalize()

    def compile_partition(self):
        self.compiled_partition = self.partition.compile(
            self.input_logical_tensors, self.output_logical_tensors, self.engine
        )
        self.kernel = self.compile_partition

    def create_partition(self):
        partitions = self.graph.get_partitions(onednn_graph.partition.fusion)
        assert len(partitions) == 1
        self.partition = partitions[0]
        return self.partition

    def execute_partition(self, input_tensors):
        execution_inputs = []
        execution_outputs = []
        for each_lt, each_tensor in zip(self.input_logical_tensors, input_tensors):
            execution_inputs.append(
                onednn_graph.tensor(each_lt, self.engine, each_tensor.data_ptr())
            )
        if not self.output_tensors:
            # created once & cached
            self.output_tensors = self.create_output_tensors()
        for each_lt, each_tensor in zip(
            self.output_logical_tensors, self.output_tensors
        ):
            execution_outputs.append(
                onednn_graph.tensor(each_lt, self.engine, each_tensor.data_ptr())
            )
        self.compiled_partition.execute(
            self.stream, execution_inputs, execution_outputs
        )
        if len(self.output_tensors) == 1:
            return self.output_tensors[0]
        else:
            return self.output_tensors

    def create_logical_tensor_from_tensor(
        self, input, property_type=onednn_graph.logical_tensor.property_type.variable
    ):
        retVal = onednn_graph.logical_tensor(
            self.lt_idx,
            onednn_graph_dtype_map[input.dtype],
            input.size(),
            input.stride(),
            property_type,
        )
        self.lt_idx += 1
        return retVal

    def create_logical_tensor(
        self,
        dtype,
        sizes,
        strides,
        property_type=onednn_graph.logical_tensor.property_type.variable,
    ):
        retVal = onednn_graph.logical_tensor(
            self.lt_idx, onednn_graph_dtype_map[dtype], sizes, strides, property_type
        )
        self.lt_idx += 1
        return retVal

    def rename(self, new_name):
        self.__name__ = new_name

    def name(self):
        return self.__name__

    @record_function("OnednnGraphPartitionModule__call__")
    def __call__(self, input_tensors):
        if getattr(input_tensors[0], "fake_mode", None):
            fake_output_tensors = self.create_output_tensors()
            return (
                fake_output_tensors[0]
                if len(fake_output_tensors) == 1
                else fake_output_tensors
            )
        return self.execute_partition(input_tensors)
