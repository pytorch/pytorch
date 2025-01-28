import dataclasses
import operator
import random
import textwrap
import types
from typing import Callable

from triton.runtime.jit import JITFunction

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .. import config, ir
from torch._inductor.runtime.triton_heuristics import grid
from .common import (
    IndentedBuffer,
)
from .wrapper import (
    PythonWrapperCodegen,
    Line,
    BufferLike,
    CommentLine,
    MemoryPlanningLine,
    MemoryPlanningState,
    EnterDeviceContextManagerLine,
    ExitDeviceContextManagerLine,
    EnterSubgraphLine,
    ExitSubgraphLine,
    AllocateLine,
    FreeLine,
    FreeIfNotReusedLine,
    ReuseLine,
    CommBufferLine,
    NullLine,
    CommBufferAllocateLine,
    CommBufferFreeLine,
)
from ..utils import (
    LineContext,
)

def call_triton_kernel(kernel: Callable, grid, args):
    """
    Call Triton kernels, for testing purposes.
    """
    return kernel[grid](args)

"""
Extra wrapper IR nodes for FX codegen.
"""
@dataclasses.dataclass
class WrapperIRLine(MemoryPlanningLine):
    """
    Base class for Wrapper IR nodes that do not participate in memory planning.
    Records the call args of the underlying codegen function.
    """
    args: tuple
    kwargs: dict
    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        pass

    def codegen(self, code: IndentedBuffer) -> None:
        raise NotImplementedError("Python codegen not supported")

class KernelCallLine(WrapperIRLine):
    pass
class KernelDefinitionLine(WrapperIRLine):
    pass

class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Generate Wrapper FX IR, for use in other backends.
    """
    def __init__(self):
        super().__init__()
        self.graph = torch.fx.Graph() # Wrapper FX IR.
        self.buffer_to_node: dict[MemoryPlanningLine, torch.fx.Node] = {} # Symbol table for codegen.
        kernels = {} # Table to store Triton kernels.

    @staticmethod
    def create(
        is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        return WrapperFxCodegen()

    def _import_kernel(kernel_name: str, code: str) -> types.ModuleType:
        """
        Imports a kernel as a python module.
        """
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            kernel.dump(path)
            spec = importlib.util.spec_from_file_location(kernel_name, f.name)

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod

    def define_kernel(
        self, *args, **kwargs,
    ):
        """
        Generates Wrapper IR for a kernel definition.
        """
        self.writeline(KernelDefinitionLine(self, args, kwargs))

    def _fake_tensor(self, size, stride, dtype) -> torch.Tensor:
        with V.fake_mode:
            return torch.empty_strided(
                size,
                stride,
                dtype=tensor_meta.dtype,
            )

    def _create_meta_from_buffer(self, node: torch.fx.Node, buffer: BufferLike) -> None:
        node.meta["val"] = self._fake_tensor(tuple(buffer.get_size()), tuple(buffer.get_stride()), dtype=buffer.get_dtype(), device=buffer.get_device())

    def _record_allocation(buffer: BufferLike, node: torch.fx.Node) -> None:
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
        assert node not in self.buffer_to_node
        self.buffer_to_node[buffer] = node

    def _free(buffer: BufferLike) -> None:
        """
        Generates FX IR to delete a buffer.
        Removes the buffer from the symbol table.
        """
        node = self.buffer_to_node[buffer]
        self.graph.call_function(operator.delitem, args=(node, None))
        del self.buffer_to_node[buffer]

    def _generate(self, is_inference):

        # We disable planning during training because it presently increases peak memory consumption.
        #TODO don't duplicate this code. Refactor into a helper in the base class.
        if is_inference and config.memory_planning:
            self.memory_plan()
        else:
            self.memory_plan_reuse()

        # Generate FX IR from Wrapper IR.
        for line in self.lines:

            line_type = type(line)
            conversion_func = {
                AllocateLine: self._generate_allocate,
                CommentLine: self._generate_comment,
                EnterDeviceContextManagerLine: self._generate_enter_device_context_manager,
                ExitDeviceContextManagerLine: self._generate_exit_device_context_manager,
                EnterSubgraphLine: self._generate_enter_subgraph,
                ExitSubgraphLine: self._generate_exit_subgraph,
                FreeIfNotReusedLine: self._generate_free_if_not_reused,
                LineContext: self._generate_line_context,
                ReuseLine: self._generate_reuse,
                NullLine: self._generate_null,
                CommBufferLine: self._generate_comm_buffer,
                CommBufferAllocateLine: self._generate_comm_buffer_allocate,
                CommBufferFreeLine: self._generate_comm_buffer_free,
                KernelDefinitionLine: self._generate_kernel_definition,
                KernelCallLine: self._generate_kernel_call,
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

            conversion_func(line)

    def _generate_allocate(self, line: Line) -> None:
        assert isinstance(line, AllocateLine)
        buffer = self.node
        name = buffer.get_name()
        assert name not in V.graph.removed_buffers

        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())

        node = self.graph.call_function(torch.empty_strided, args=(shape, stride, dtype, device))
        node.name = name
        self._create_meta_from_buffer(node, buffer)
        self._record_allocation(buffer, node)

    def _generate_comment(self, line: Line) -> None:
        assert isintance(line, CommentLine)
        # We ignore comments in FX IR.

    def _generate_enter_device_context_manager(self, line: Line) -> None:
        assert isinstance(line, EnterDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_exit_device_context_manager(self, line: Line) -> None:
        assert isinstance(line, ExitDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_enter_subgraph(self, line: Line) -> None:
        assert isinstance(line, EnterSubgraphLine)
        raise NotImplementedError("Subgraphs are not yet supported by FX conversion")

    def _generate_exit_subgraph(self, line: Line) -> None:
        assert isinstance(line, ExitSubgraphLine)
        raise NotImplementedError("Subgraphs are not yet supported by FX conversion")

    def _generate_free_if_not_reused(self, line: Line) -> None:
        assert isinstance(line, FreeIfNotReusedLine)
        buf = line.node
        assert buf.get_name() not in V.graph.removed_buffers
        if not buf.is_reused:
            buf._free(self.node)

    def _generate_line_context(self, line: Line) -> None:
        assert isinstance(line, LineContext)
        # We ignore line context in FX IR.

    def _generate_reuse(self, line: Line) -> None:
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
            self._create_meta_from_buffer(result_node, new)

        self._record_allocation(new, result_node)

    def _generate_null(self, line: Line) -> None:
        assert isintstance(line, NullLine)
        # Does nothing.

    def _generate_comm_buffer(self, line: Line) -> None:
        assert isinstance(line, CommBufferLine)
        # Does nothing. Comm buffers are handled by the respective allocate/free lines.

    def _generate_comm_buffer_allocate(self, line: Line) -> None:
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
        from torch._C._distributed_c10d._SymmetricMemory import empty_strided_p2p
        alloc_id = random.randint(0, 2**64 - 1)
        node = self.graph.call_function(
                empty_strided_p2p,
                args=(shape, stride, dtype, device, self.group_name, alloc_id),
        )
        self._create_meta_from_buffer(result_node, buf)
        self._record_allocation(buf, node)

    def _generate_comm_buffer_free(self, line: Line) -> None:
        assert isinstance(line, CommBufferFreeLine)
        self.graph.free(line.node)

    def _generate_triton_call(self, line: Line) -> None:
        assert isinstance(line, KernelCallLine) #TODO create this in Wrapper IR

        if line.kwargs["grid_fn"] not in ("grid", None):
            raise NotImplementedError(f"Unsupported grid_fn: '{grid_fn}'")


        kernel_name, call_args = line.args
        kernel = self.kernels[kernel_name]

        node = self.graph.call_function(call_triton_kernel, args=(kernel, grid, call_args))

    def generate_kernel_call(
        self, *args, **kwargs,
    ):
        """
        Generates Wrapper IR for a kernel call.
        """
        self.writeline(KernelCallLine(self, args, kwargs))

    def _generate_kernel_call(self, line: Line):
        assert isinstance(line, KernelCallLine)
        if not line.kwargs["triton"]:
            raise NotImplementedError("FX conversion only supports Triton kernels.")

        self._generate_triton_call(line)


    def _generate_kernel_definition(self, line: Line):
        assert isinstance(line, KernelDefinitionLine)

        # Generate code for the kernel.
        #TODO refactor into parent class?
        kernel_name, kernel_body = line.args
        metadata = line.kwargs["metadata"]
        have_metadata = metadata and not config.triton.autotune_at_compile_time
        metadata_comment = f"{metadata}\n" if have_metadata else ""
        kernel_code = f"\n\n{metadata_comment}{kernel_name} = {kernel_body}"

        # Import the code and store the kernel.
        mod = self._import_kernel(kernel_name, kernel_code)
        kernel = getattr(mod, kernel_name)
        self.kernels[kernel_name] = kernel
