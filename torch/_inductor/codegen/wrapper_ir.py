import torch
import operator
import random

from .. import config, ir

BufferLike = Union[ir.Buffer, WorkspaceArg]

"""
Wrapper IR functions.
"""
# torch.empty_strided -- allocs
# reinterpret_view -- reuse
# operator.delitem -- del

"""
Memory planning IR. These generate wrapper IR with self.codegen.
"""
@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

    def codegen(self, graph: torch.fx.Graph) -> None:
        """Second pass to output wrapper IR"""

    def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        # TODO move this to python wrapper.py. This is Python only.
        args: List[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: BufferLike

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        if self.node.get_device_or_error().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, ir: WrapperIR) -> None:
        name = self.node.get_name()
        assert name not in V.graph.removed_buffers

        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        node = ir.graph.call_function(torch.empty_strided, args=(shape, stride, dtype, device))
        node.name = name

        ir.record_allocation(buffer, node)


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: BufferLike
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, ir: WrapperIR) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            ir.free(self.node)


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: BufferLike
    reused_as: BufferLike
    delete_old: bool = True

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, ir: WrapperIR) -> None:
        old = self.node
        new = self.reused_as
        assert not any(buf.get_name() in V.graph.removed_buffers for buf in (old, new))
        assert old.get_dtype() == new.get_dtype()

        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            # No operation necessary. Simply update the symbol table.
            ir.record_allocation(new, ir.buffer_to_node[old])
        else:
            return
            reinterpret_view = self.codegen_reinterpret_view(
                old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
            )
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        if old_name not in V.graph.get_output_names() and delete_old:
            ir.free(old)


class NullLine(MemoryPlanningLine):
    pass


@dataclasses.dataclass
class CommBufferLine(WrapperLine):
    wrapper: PythonWrapperCodegen  # type: ignore[name-defined] # noqa: F821
    node: ir.Buffer

    @property
    def size(self) -> int:
        from torch._inductor.utils import is_symbolic

        numel = self.node.get_numel()
        dtype = self.node.get_dtype()
        if is_symbolic(numel):
            raise AssertionError(
                f"The size of a comm buffer can't be symbolic: {self.node}"
            )
        return int(numel) * dtype.itemsize

    @property
    def comm_buffer_type(self) -> ir.CommBufferType:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.comm_buffer_type

    @property
    def group_name(self) -> str:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.group_name


@dataclasses.dataclass
class CommBufferAllocateLine(CommBufferLine):
    def codegen(self, ir: WrapperIR) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        name = self.node.get_name()
        device = torch.device(f"cuda:{self.node.get_device().index}")
        dtype = self.node.get_dtype()
        shape = tuple(self.node.get_size())
        stride = tuple(self.node.get_stride())

        if self.comm_buffer_type != ir.CommBufferType.SYMM_MEM:
            raise NotImplementedError(
                f"Unsupported comm buffer type: {comm_buffer_type}"
            )

        # Distributed is not always avaliable. Only import it if used.
        from torch._C._distributed_c10d import _SymmetricMemory import empty_strided_p2p
        alloc_id = random.randint(0, 2**64 - 1)
        node = ir.graph.call_function(empty_strided_p2p, shape, stride, dtype, device, self.group_name, alloc_id)
        ir.record_allocation(self.node, node)


@dataclasses.dataclass
class CommBufferFreeLine(CommBufferLine):
    def codegen(self, ir: WrapperIR) -> None:
        ir.free(self.node)


BufferName = str


class CallKernelLine(MemoryPlanningLine):
    #TODO generate this in codegen_kernel_call. Store kernel info in a structured way.
    def codegen(self, graph: torch.fx.Graph):
        graph.call_function(call_kernel)

class WrapperIR():
    def __init__(self):
        self.graph = torch.fx.Graph() # Wrapper FX IR.
        self.buffer_to_node: Dict[MemoryPlanningLine, torch.fx.Node] = {} # Symbol table for codegen.

    def record_allocation(buffer: BufferLike, node: torch.fx.Node):
        assert node not in self.buffer_to_node
        self.buffer_to_node[buffer] = node

    def free(buffer: BufferLike):
        node = self.buffer_to_node[buffer]
        ir.graph.call_function(operator.delitem, args=(node, None))
        del self.buffer_to_node[buffer]

class WrapperIRCodegen(CodeGen):
    """
    Generate Wrapper IR, for use in other backends.
    """
    def __init__(self):
        super().__init__()
        self._names_iter: Iterator[int] = count()
        self.lines: List[Union[MemoryPlanningLine]] = [] # Memory planning IR. Used as an intermediate step.
        self.wrapper_ir = WrapperIR() # Wrapper IR. The output of this pass.

        self.allocated = OrderedSet[BufferName]()
        self.freed = OrderedSet[BufferName]()

        # maps from reusing buffer to reused buffer
        self.reuses: Dict[BufferName, BufferName] = {}

    def generate(self, is_inference):
        with dynamo_timed("WrapperIRCodegen.generate"):
            return self._generate(is_inference)

    def _generate(self, is_inference):

        # We disable planning during training because it presently increases peak memory consumption.
        if is_inference and config.memory_planning:
            self.memory_plan()
        else:
            self.memory_plan_reuse()

        # Codegen wrapper IR from memory planning IR.
        for line in self.lines:
            #FIXME: in the original codegen, this could also include strings,
            # which were pasted directly into the code.
            # Now, everything has to have an FX IR equivalent.
            line.codegen(self.wrapper_ir)

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        assert len(planning_states) == 0

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        # FIXME(rec): not used
        _total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

    def did_reuse(self, buffer, reused_buffer):
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer: ir.Buffer, output_buffer: ir.Buffer):
        assert can_match_buffer_size(input_buffer, output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

