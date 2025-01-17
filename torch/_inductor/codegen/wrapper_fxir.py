import torch
import operator
import random
import textwrap

from .. import config, ir
from .wrapper import (
    PythonWrapperCodegen,
    Line,
    AllocateLine,
    FreeIfNotReusedLine,
    ReuseLine,
    CommBufferLine,
    NullLine,
    CommBufferAllocateLine,
    CommBufferFreeLine,
)

class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Generate Wrapper FX IR, for use in other backends.
    """
    def __init__(self):
        super().__init__()
        self.graph = torch.fx.Graph() # Wrapper FX IR.
        self.buffer_to_node: Dict[MemoryPlanningLine, torch.fx.Node] = {} # Symbol table for codegen.

    def _record_allocation(buffer: BufferLike, node: torch.fx.Node):
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
        assert node not in self.buffer_to_node
        self.buffer_to_node[buffer] = node

    def _free(buffer: BufferLike):
        """
        Generates FX IR to delete a buffer.
        Removes the buffer from the symbol table.
        """
        node = self.buffer_to_node[buffer]
        self.graph.call_function(operator.delitem, args=(node, None))
        del self.buffer_to_node[buffer]

    def _generate(self, is_inference):

        # We disable planning during training because it presently increases peak memory consumption.
        if is_inference and config.memory_planning:
            self.memory_plan()
        else:
            self.memory_plan_reuse()

        # Generate FX IR from Wrapper IR.
        for line in self.lines:

            line_type = type(line)
            conversion_func = {
                AllocateLine: _generate_allocate,
                FreeIfNotReusedLine: _generate_free_if_not_reused,
                ReuseLine: _generate_reuse,
                NullLine: _generate_null,
                CommBufferLine: _generate_comm_buffer,
                CommBufferAllocateLine: _generate_comm_buffer_allocate,
                CommBufferFreeLine: _generate_comm_buffer_free,
            }.get(line_type)

            # FX conversion only supports Wrapper IR, not Python/C++ lines.
            if conversion_func is None:
                raise NotImplementedError(textwrap.dedent(
                    f"""
                    Found line of unrecognized type '{line_type}':
                        '{line}'

                    FX conversion only supports Wrapper IR lines.
                    """
                ))

            conversion_func(self, line)

    def _generate_allocate(self, line: Line):
        assert isinstance(line, AllocateLine)
        name = self.node.get_name()
        assert name not in V.graph.removed_buffers

        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        node = self.graph.call_function(torch.empty_strided, args=(shape, stride, dtype, device))
        node.name = name

        self._record_allocation(buffer, node)

    def _generate_free_if_not_reused(self, line: Line):
        assert isinstance(line, FreeIfNotReusedLine)
        buf = line.node
        assert buf.get_name() not in V.graph.removed_buffers
        if not buf.is_reused:
            buf._free(self.node)

    def _generate_reuse(self, line: Line):
        assert isinstance(line, ReuseLine)
        old = line.node
        new = line.reused_as
        assert not any(buf.get_name() in V.graph.removed_buffers for buf in (old, new))
        assert old.get_dtype() == new.get_dtype()

        result_node = self.buffer_to_node[old]

        # Free the old buffer.
        if old.get_name() not in V.graph.get_output_names() and delete_old:
            self._free(old)

        # Change shape and stride.
        size = new.get_size()
        stride = new.get_stride()
        offset = old.offset()
        if old.get_size() != size or old.get_stride() != stride or old.offset() != offset:
            result_node = self.graph.call_function(
                torch.as_strided,
                args=(size, stride, offset)
            )

        self._record_allocation(new, result_node)

    def _generate_null(self, line: Line):
        assert isintstance(line, NullLine)
        # Does nothing.

    def _generate_comm_buffer(self, line: Line):
        assert isinstance(line, CommBufferLine)
        # Does nothing. Comm buffers are handled by the respective allocate/free lines.

    def _generate_comm_buffer_allocate(self, line: Line):
        assert isinstance(line, CommBufferFreeLine)
        buf = line.node
        assert buf.get_name() not in V.graph.removed_buffers
        name = bug.get_name()
        device = torch.device(f"cuda:{self.node.get_device().index}")
        dtype = buf.get_dtype()
        shape = tuple(buf.get_size())
        stride = tuple(buf.get_stride())

        if self.comm_buffer_type != ir.CommBufferType.SYMM_MEM:
            raise NotImplementedError(
                f"Unsupported comm buffer type: {comm_buffer_type}"
            )

        # Distributed is not always avaliable. Only import it if used.
        from torch._C._distributed_c10d import _SymmetricMemory import empty_strided_p2p
        alloc_id = random.randint(0, 2**64 - 1)
        node = self.graph.call_function(empty_strided_p2p, shape, stride, dtype, device, self.group_name, alloc_id)
        self._record_allocation(buf, node)

    def _generate_comm_buffer_free(self, line: Line):
        assert isinstance(line, CommBufferFreeLine)
        self.graph.free(line.node)

