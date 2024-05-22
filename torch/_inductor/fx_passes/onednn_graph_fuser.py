from typing import Dict

import torch

# This module is imported only if oneDNN Graph can be used
import torch._C._onednn_graph as onednn_graph  # type: ignore[import-not-found]
from torch.profiler import record_function
from ..ir import InductorImplicitFallback

onednn_graph_dtype_map: Dict[torch.dtype, onednn_graph.logical_tensor] = {
    torch.float16: onednn_graph.logical_tensor.f16,
    torch.bfloat16: onednn_graph.logical_tensor.bf16,
    torch.float32: onednn_graph.logical_tensor.f32,
    torch.int32: onednn_graph.logical_tensor.s32,
    torch.int8: onednn_graph.logical_tensor.s8,
    torch.uint8: onednn_graph.logical_tensor.u8,
    torch.bool: onednn_graph.logical_tensor.boolean,
}

onednn_graph_reverse_dtype_map: Dict[onednn_graph.logical_tensor, torch.dtype] = {
    onednn_graph_dtype_map[key]: key for key in onednn_graph_dtype_map
}


class OnednnGraphPartitionModule(InductorImplicitFallback):
    """Creates a python callable object corresponding to a fused kernel.

    In an Inductor pass, a subgraph of the whole graph may be replaced by a single node
    corresponding to a fusion-pattern supported by oneDNN Graph.
    Currently, these fusion patterns are hardcoded in oneDNN Graph.
    API reference is at https://spec.oneapi.io/versions/latest/elements/oneDNN/source/graph/programming_model.html
    """

    def __init__(self, name):
        super().__init__()
        self.is_opaque = True
        self.__name__ = name  # type: ignore[assignment]
        self.partition: onednn_graph.partition = None
        self.compiled_partition: onednn_graph.compiled_partition = None
        self.kernel = None
        self.tags = []
        # cache output tensors to reuse their storage
        self.output_tensors = []
        self.namespace = "onednn_graph"

        # oneDNN Graph specific data structures
        # API reference:
        self.graph = onednn_graph.graph(onednn_graph.engine.cpu)
        self.engine = onednn_graph.engine(onednn_graph.engine.cpu, 0)
        self.stream = onednn_graph.stream(self.engine)
        self.lt_idx = 0
        self.op_idx = 0
        self.partitions = None
        self.input_logical_tensors = []
        self.output_logical_tensors = []

    def add_op(self, op: onednn_graph.op):
        """Add an op to oneDNN Graph."""
        self.graph.add_op(op, True)

    def create_output_tensors(self):
        """
        Allocate PyTorch output tensors

        Args: None

        Returns: output tensors of a oneDNN Graph partition
        """
        retVal = []
        for output_tensor in self.output_logical_tensors:
            retVal.append(
                torch.empty_strided(
                    output_tensor.get_dims(), output_tensor.get_strides()
                ).to(onednn_graph_reverse_dtype_map[output_tensor.get_data_type()])
            )
        return retVal

    def create_op(self, op_name, op_str):
        """Create a oneDNN Graph op, which is subsequently added to a oneDNN Graph graph with add_op."""
        op = onednn_graph.op(self.op_idx, op_name, op_str)
        self.op_idx += 1
        return op

    def add_input_logical_tensor(self, input_tensor: onednn_graph.logical_tensor):
        """Append oneDNN Graph input logical tensors to a list, which is cached"""
        self.input_logical_tensors.append(input_tensor)

    def add_output_logical_tensor(self, output_tensor: onednn_graph.logical_tensor):
        """Append a oneDNN Graph output logical tensor to a list, which is cached"""
        self.output_logical_tensors.append(output_tensor)

    def finalize_graph(self):
        """Using this API is necessary before getting partitions"""
        self.graph.finalize()

    def compile_partition(self):
        """JIT compiles & caches fused kernel"""
        self.compiled_partition = self.partition.compile(
            self.input_logical_tensors, self.output_logical_tensors, self.engine
        )
        self.kernel = self.compile_partition

    def create_partition(self):
        """Creates a partition (fused kernel) corresponding to a hardcoded pattern"""
        partitions = self.graph.get_partitions(onednn_graph.partition.fusion)
        assert len(partitions) == 1
        self.partition = partitions[0]
        return self.partition

    def execute_partition(self, input_tensors):
        """Executes a fused kernel corresponding to a hardcoded pattern"""
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
        """Creates a logical tensor (oneDNN Graph counterpart of a FakeTensor from a PyTorch tensor)"""
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
        dtype: torch.dtype,
        sizes: torch.Size,
        strides: tuple[int, ...],
        property_type=onednn_graph.logical_tensor.property_type.variable,
    ):
        """Creates a logical tensor (oneDNN Graph counterpart of a FakeTensor)"""
        retVal = onednn_graph.logical_tensor(
            self.lt_idx, onednn_graph_dtype_map[dtype], sizes, strides, property_type
        )
        self.lt_idx += 1
        return retVal

    def name(self):
        return self.__name__

    @record_function("OnednnGraphPartitionModule__call__")
    def __call__(self, input_tensors):
        """
        The callable object is called & the output of the fused kernel is returned

        Args:
            input_tensors: a tuple of input tensors

        Returns:
            output of fused kernel ("partition" in oneDNN Graph lexicon)
        """
        if getattr(input_tensors[0], "fake_mode", None):
            fake_output_tensors = self.create_output_tensors()
            return (
                fake_output_tensors[0]
                if len(fake_output_tensors) == 1
                else fake_output_tensors
            )
        return self.execute_partition(input_tensors)
