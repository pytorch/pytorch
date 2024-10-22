# mypy: disallow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import pprint
import textwrap
import traceback
import typing
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Counter,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import free_symbol_is_type, SymT
from torch.utils._triton import has_triton

from . import comms, config, dependencies, ir, metrics
from .codecache import write_text
from .codegen.common import BackendFeature, get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import Dep, MemoryDep, StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .loop_body import LoopBody
from .memory import MemoryPlanningInfoForBuffer, MemoryPlanningInfoForNode
from .runtime.runtime_utils import green_text, red_text
from .sizevars import SimplifyIndexing
from .utils import (
    cache_on_self,
    cmp,
    device_need_guard,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    IndentedBuffer,
    is_collective,
    is_gpu,
    is_wait,
    sympy_product,
)
from .virtualized import V


log = logging.getLogger(__name__)
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
loop_ordering_log = torch._logging.getArtifactLogger(__name__, "loop_ordering")


@dataclasses.dataclass
class SchedulerBuffer:
    scheduler: Scheduler
    node: ir.Buffer
    defining_op: BaseSchedulerNode
    users: List[NodeUser] = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def __hash__(self) -> int:
        return hash(self.node.name)

    def debug_str(self) -> str:
        result = IndentedBuffer()
        name = self.get_name()
        result.writeline(f"{name}: {type(self.node).__name__}")
        result.writeline(f"{name}.layout = {self.node.layout}")
        if self.get_aliases():
            result.writeline(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            result.writeline(f"{name}.mutations = {pformat(self.get_mutations())}")

        if len(self.users) <= 1:
            result.writeline(f"{name}.users = {self.users}")
        else:
            result.writeline(f"{name}.users = [")
            with result.indent(1):
                for user in self.users:
                    result.writeline(f"{user},")
            result.writeline("]")
        return result.getrawvalue()

    def get_name(self) -> str:
        return self.node.get_name()

    def allocate(self) -> None:
        assert self.node is not None
        if not self.node.should_allocate():
            return

        if self.node.get_inputs_that_alias_output() or self.node.get_mutation_names():
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # hacky check for if V.kernel is a real kernel or NullHandler
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            V.graph.wrapper_code.codegen_inplace_reuse(
                self.scheduler.name_to_buf[
                    V.kernel.inplace_update_buffers[self.get_name()]
                ].node,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self) -> bool:
        # There's no real allocated buffer, no need to free it
        assert self.node is not None
        if isinstance(self.node.layout, ir.NoneLayout):
            return False
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def set_users(self, users: List[NodeUser]) -> None:
        # deduplicate
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self) -> Sequence[str]:
        assert self.node is not None
        return self.node.get_inputs_that_alias_output()

    def get_mutations(self) -> List[str]:
        assert self.node is not None
        return self.node.get_mutation_names()


class BaseSchedulerNode:
    group: Tuple[torch.device, Tuple[Tuple[sympy.Expr, ...], ...]]
    read_writes: dependencies.ReadWrites
    unmet_dependencies: OrderedSet[Dep]
    # .min_order and .max_order are only relevant for "grouped" nodes such as FusedSchedulerNode.
    # e.g. if the FusedSchedulerNode includes nodes (op_1, op_2, op_3), and op_X is X-th node
    # in `self.scheduler.nodes`, then for this FusedSchedulerNode, .min_order is 1 and .max_order is 3.
    # For non-"grouped" nodes (i.e. regular SchedulerNode),
    # .min_order = .max_order = X if this node is X-th node in `self.scheduler.nodes`.
    min_order: int
    max_order: int
    mpi_node: MemoryPlanningInfoForNode

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler: Scheduler = scheduler
        self.debug_device_str: Callable[
            [BaseSchedulerNode], List[str]
        ] = lambda *args, **kwargs: []

    def _init_from_node(self, node: ir.Operation) -> None:
        self.node: Optional[ir.Operation] = node
        self.ancestors: OrderedSet[str] = OrderedSet()
        self.last_usage: OrderedSet[
            str
        ] = OrderedSet()  # buffers that won't be used after this kernel
        self.written = False
        self.outputs: List[SchedulerBuffer] = [
            SchedulerBuffer(
                scheduler=self.scheduler,
                node=output,
                defining_op=self,
            )
            for output in node.get_outputs()
        ]
        self.outputs_by_name: Dict[str, SchedulerBuffer] = {
            buf.get_name(): buf for buf in self.outputs
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({type(getattr(self, 'node', None)).__name__})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.outputs = [
        """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    def debug_str_extra(self) -> str:
        return ""

    def _debug_str_for_device(self) -> List[str]:
        return self.debug_device_str(self)

    def debug_str_short(self) -> str:
        maybe_data = getattr(self.node, "data", None)
        data_str = ""
        if isinstance(maybe_data, torch._inductor.ir.Pointwise):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_size()], shorten=False, multiline=False
            )
        elif isinstance(maybe_data, torch._inductor.ir.Reduction):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_reduction_size(), maybe_data.get_reduction_type()],
                shorten=False,
                multiline=False,
            )
        return f"{self}{data_str}"

    def log_details(self) -> None:
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> None:
        return

    def update_mutated_names(self, renames: Dict[str, str]) -> None:
        self.set_read_writes(self.read_writes.rename(renames))

    def add_fake_dep(self, dep: Dep) -> None:
        self.set_read_writes(self.read_writes.with_read(dep))

    def has_aliasing_or_mutation(self) -> bool:
        return any(
            buf.get_aliases() or buf.get_mutations() for buf in self.get_outputs()
        )

    def set_read_writes(self, rw: dependencies.ReadWrites) -> None:
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
    ) -> None:
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = OrderedSet([mutation_real_name.get(k, k) for k in used_buffers])
        self.last_usage = used_buffers - future_used_buffers

    def mark_run(self) -> None:
        for buf in self.outputs:
            buf.allocate()

    def used_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet(
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        )

    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        used_names: OrderedSet[str] = OrderedSet()

        deps = [
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        ]
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            if V.graph.name_to_buffer.get(dep):
                for alias in V.graph.name_to_buffer[dep].get_inputs_that_alias_output():
                    if alias not in used_names:
                        deps.append(alias)
        return used_names

    def prune_deps(self) -> None:
        self.unmet_dependencies = OrderedSet(
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        )

    def prune_weak_deps(self) -> None:
        # Prune weak dependencies on operations that have been removed
        def should_prune(dep: Dep) -> bool:
            if not isinstance(dep, WeakDep):
                return False
            op = self.scheduler.name_to_buf[dep.name].defining_op
            return op.get_name() in V.graph.removed_operations

        to_remove = OrderedSet(
            dep for dep in self.read_writes.reads if should_prune(dep)
        )
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(
        self, name_to_fused_node: Dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

    def get_name(self) -> str:
        assert self.node is not None
        return self.node.get_operation_name()

    def get_first_name(self) -> str:
        return self.get_name()

    @cache_on_self
    def get_operation_names(self) -> OrderedSet[str]:
        return OrderedSet([node.get_name() for node in self.get_nodes()])

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet([out.get_name() for out in self.outputs])

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return [self]

    def get_outputs(self) -> Sequence[SchedulerBuffer]:
        return self.outputs

    def get_output(self, buf_name: str) -> SchedulerBuffer:
        return self.outputs_by_name[buf_name]

    def get_device(self) -> torch.device:
        assert self.node is not None
        return self.node.get_device()

    def is_reduction(self) -> bool:
        return False

    def is_split_scan(self) -> bool:
        return False

    def is_template(self) -> bool:
        return False

    def is_extern(self) -> bool:
        return False

    def is_foreach(self) -> bool:
        return False

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return False

    def decide_inplace_update(self) -> None:
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
        from .codegen.wrapper import buffer_reuse_key

        if not (
            isinstance(self, (SchedulerNode,))
            and config.inplace_buffers
            and V.graph.has_feature(self.get_device(), BackendFeature.INPLACE_BUFFERS)
            and (
                not isinstance(V.kernel, torch._inductor.codegen.simd.SIMDKernel)
                or getattr(V.kernel, "mutations", None) is not None
            )
            # hacky check for if V.kernel is a real kernel or NullHandler
            and hasattr(V.kernel, "args")
        ):
            return
        fused_nodes = {
            node.get_name()
            for node in self.scheduler.name_to_fused_node[self.get_name()].get_nodes()
        }

        ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)

        for buf in self.get_outputs():
            buf_node = buf.node
            assert buf_node is not None
            if (
                not buf_node.should_allocate()
                or buf_node.get_inputs_that_alias_output()
                or buf_node.get_mutation_names()
                or buf.get_name() in V.graph.removed_buffers
            ):
                continue

            for read in ordered_reads:
                input_buf: Optional[SchedulerBuffer] = self.scheduler.name_to_buf.get(
                    read.name
                )
                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, self)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    # If the writers of input_buf are in the same FusedSchedulerNode as the current op, then there is
                    # no need to inplace.
                    if input_buf.defining_op.get_name() in fused_nodes:
                        continue

                    assert input_buf.users is not None
                    remaining_uses = [
                        x
                        for x in input_buf.users
                        if x.node.get_name() not in self.scheduler.completed_operations
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and input_buf.node is not None
                        and not isinstance(
                            input_buf.node.get_layout(),
                            (
                                ir.MultiOutputLayout,
                                ir.MutationLayoutSHOULDREMOVE,
                            ),
                        )
                        and not (
                            isinstance(
                                input_buf.defining_op.node,
                                (ir.FallbackKernel, ir.MultiOutput),
                            )
                            and len(input_buf.node.get_inputs_that_alias_output()) > 0
                        )
                        and buffer_reuse_key(input_buf.node)
                        == buffer_reuse_key(buf.node)
                    ):
                        # if there isn't a triton kernel, then we don't need to call triton-specific things.
                        # but TODO this might be a convenient place to signal to the Collective kernels to inplace
                        # (and, can we make "kernel" less generic of a name?)
                        V.kernel.args.make_inplace(input_buf.get_name(), buf.get_name())
                        # mutations not tracked in cpp kernels
                        if isinstance(
                            V.kernel, torch._inductor.codegen.simd.SIMDKernel
                        ):
                            V.kernel.mutations.add(input_buf.get_name())
                            V.kernel.mutations.add(buf.get_name())

                        # update last usage of reused node
                        self.last_usage.discard(input_buf.get_name())

                        V.kernel.inplace_update_buffers[
                            buf.get_name()
                        ] = input_buf.get_name()
                        break

    def codegen_originating_info(
        self, buffer: IndentedBuffer, only_once: bool = True
    ) -> None:
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        assert self.node is not None
        origins = self.node.get_origins()
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are boring and samey
                continue

            out_lines.append("")
            # TODO(voz): Should the pragma be constant somewhere?
            out_lines.append("#pragma CMT ORIGIN:")
            op_info_str = f"#pragma CMT {o.op} {o.target}"
            if "seq_nr" in o.meta:
                op_info_str = op_info_str + f" seq_nr:{o.meta['seq_nr']}"
            out_lines.append(op_info_str)
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.split("|")[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        buffer.writelines(out_lines)
        self.written = True

    @cache_on_self
    def get_read_write_buffers_sizes(self) -> int:
        """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size
        """
        if isinstance(self, NopKernelSchedulerNode):
            return 0
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(
            self.node, MultiOutput
        ):
            # todo: Calculate this - it's kinda annoying.
            return 0

        def try_size_hint(s: sympy.Expr) -> int:
            return V.graph.sizevars.size_hint(s, fallback=0)

        if isinstance(self, SchedulerNode):
            node_numel = try_size_hint(
                sympy_product(self.get_ranges()[0])
                * sympy_product(self.get_ranges()[1]),
            )
        else:
            node_numel = int(1e9)
        buf_accesses = collections.defaultdict(list)
        for dep in self.read_writes.reads | self.read_writes.writes:
            buf_accesses[dep.name].append(dep)

        reads = OrderedSet(dep.name for dep in self.read_writes.reads)
        writes = OrderedSet(dep.name for dep in self.read_writes.writes)

        def is_materialized(buf: str, snodes: Sequence[BaseSchedulerNode]) -> bool:
            users = self.scheduler.name_to_buf[buf].users
            buf_uses = OrderedSet(user.node for user in users)
            return len(buf_uses - OrderedSet(snodes)) > 0

        if isinstance(self, FusedSchedulerNode):
            removed_buffers = OrderedSet(
                dep for dep in writes if not is_materialized(dep, self.snodes)
            )
            writes = writes - removed_buffers
            reads = reads - removed_buffers
        node_bytes = 0

        for buf_name in reads | writes:
            buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
            buf: Union[ir.Buffer, ir.TensorBox]
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_bytes(buf: Optional[Union[ir.Buffer, ir.TensorBox]]) -> int:
                if not buf:
                    return 0
                # Kind of a lazy way to get the MultiOutput nodes corresponding to
                # a MultiOutputLayout
                if isinstance(buf.layout, MultiOutputLayout):
                    users = self.scheduler.name_to_buf[buf.get_name()].users
                    tot = 0
                    for user in users:
                        assert isinstance(user.node, BaseSchedulerNode)
                        if isinstance(user.node.node, MultiOutput):
                            for sched_buf in user.node.get_outputs():
                                tot += get_buf_bytes(sched_buf.node)
                        else:
                            # Buf is a MultiOutputLayout but not all of its
                            # users are MultiOutputs...
                            # TODO: Figure out what's going on
                            return 0
                    return tot
                elif isinstance(buf.layout, ir.NoneLayout):
                    return sum(
                        get_buf_bytes(V.graph.get_buffer(mut_name))
                        for mut_name in buf.get_mutation_names()
                    )
                else:
                    buf_elems = try_size_hint(sympy_product(buf.get_size()))
                    return get_dtype_size(buf.get_dtype()) * min(
                        buf_accessed_elems, buf_elems
                    )

            node_bytes += get_buf_bytes(buf)

        return node_bytes

    @cache_on_self
    def get_estimated_runtime(self) -> float:
        """
        Returns estimated op runtime in nanoseconds (ns)
        """
        buf = self.get_nodes()[0].get_outputs()[0]
        layout = buf.node.get_layout()
        dtype = buf.node.get_dtype()

        if layout.device is not None and not is_gpu(layout.device.type):
            # default to no reordering based on runtime
            return 0

        # Collective kernels
        if is_collective(self.node):
            assert isinstance(self.node, ir.IRNode)
            try:
                return estimate_nccl_collective_runtime(self.node)
            except ValueError as e:
                # We don't know how to estimate runtime for this collective,
                # falling back to 0
                log.info(e)
                return 0

        elif is_wait(self.node):
            # ir.Wait is only used for collective ops.
            # The time needed for the collective op is already estimated and considered
            # when we are processing the collective op IR node, so ir.Wait takes 0 time
            # since it doesn't take extra time to get the result after the collective is completed.
            return 0

        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
        except Exception:
            return 0

        if isinstance(self, ExternKernelSchedulerNode):
            assert isinstance(self.node, ir.ExternKernel), f"{type(self.node)=}"
            op = kernel_name_to_op.get(
                getattr(self.node, "python_kernel_name", ""), None
            )

            # if there is a resolved op, dry-run using fake mode and record flop count
            if op is not None:
                from torch._subclasses.fake_tensor import FakeTensorMode
                from torch.utils.flop_counter import FlopCounterMode

                if any(
                    len(free_unbacked_symbols(n.get_numel())) > 0
                    for n in self.node.inputs
                ):
                    # Tensor has unbacked symints, we don't know how to estimate
                    # runtime for that today
                    return 0

                with FakeTensorMode() as fake_mode, FlopCounterMode(
                    display=False
                ) as flop_counter_mode, V.set_current_node(
                    self.node.fx_node
                ), V.set_fake_mode(
                    fake_mode
                ):
                    from .ir import ir_node_to_tensor

                    fake_inputs = [
                        ir_node_to_tensor(input, guard_shape=False)
                        for input in self.node.inputs
                    ]
                    cls = self.node.__class__
                    cls.process_kernel(op, *fake_inputs, **self.node.kwargs)

                    # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
                    factor = 1.0
                    counted_flops = flop_counter_mode.get_total_flops()
                    counted_bytes = self.get_read_write_buffers_sizes()
                    compute_time = (factor * counted_flops / gpu_flops) * 1e9
                    transfer_time = counted_bytes / gpu_memory_bandwidth

                    # Return estimated runtime in nanoseconds
                    return max(compute_time, transfer_time)

        elif isinstance(self, FusedSchedulerNode) or isinstance(
            self.node, ComputedBuffer
        ):
            # Return estimated runtime in nanoseconds (bytes / gbps)
            return self.get_read_write_buffers_sizes() / gpu_memory_bandwidth

        return 0

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return None


class WhyNoFuse:
    # TODO when we drop support for Python < 3.10, we can use
    # @dataclass(slots=True) instead of manually specifying __slots__.
    __slots__ = ["node1", "node2", "reason", "args"]
    reason: str
    args: Tuple[Any, ...]

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        self.node1 = node1
        self.node2 = node2

    def __call__(self, reason: str, *args: Any) -> None:
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self) -> str:
        return f"cannot fuse {self.node1.get_name()} with {self.node2.get_name()}: " + (
            self.reason % self.args
        )


def pformat(obj: Any) -> str:
    if isinstance(obj, OrderedSet):
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' ' * 4)}"
    return result


class OutputNode:
    def __init__(self, dep: StarDep) -> None:
        self.unmet_dependencies = OrderedSet([dep])

    def is_reduction(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return ()

    def get_name(self) -> str:
        return "OUTPUT"

    __repr__ = get_name


def _prune_redundant_deps(
    node: BaseSchedulerNode,
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
) -> None:
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """
    name_to_dep_count: Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep):
            op = name_to_buf[dep.name].defining_op
            name_to_dep_count[name_to_fused_node[op.get_name()].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op.get_name()
            is_redundant = name_to_dep_count[name_to_fused_node[op_name].get_name()] > 0
            # These can occur because fused nodes always gather deps from their snodes
            # If B has a weakdep on A
            # B gets fused with C, then any time BC is fused, the weakdep will reappear
            is_self_dep = name_to_fused_node[op_name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = OrderedSet(
        dep for dep in node.unmet_dependencies if should_prune(dep)
    )

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


# TODO(xmfan): reuse: an existing mapping for this if it exists, or formalize this into ir.py:ExternKernel
kernel_name_to_op = {
    "extern_kernels.convolution": torch.ops.aten.convolution,
    "extern_kernels.mm": torch.ops.aten.mm,
    "extern_kernels.bmm": torch.ops.aten.bmm,
    "extern_kernels.addmm": torch.ops.aten.addmm,
}


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())

    def debug_str_extra(self) -> str:
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'python_kernel_name', None)}"

    def is_extern(self) -> bool:
        return True

    def has_side_effects(self) -> bool:
        assert self.node is not None
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()


class NopKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())


class SchedulerNode(BaseSchedulerNode):
    def __init__(
        self,
        scheduler: Scheduler,
        node: Union[ir.ComputedBuffer, ir.TemplateBuffer],
    ) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self._compute_attrs()

    def _compute_attrs(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))
        self._sizes, self._body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )

        group_fn = self.scheduler.get_backend(self.node.get_device()).group_fn
        self.group = (self.node.get_device(), group_fn(self._sizes))

        # Don't normalize since normalization will merge loops which
        # makes it hard to decide new loop orders.
        should_normalize = (
            not config.loop_ordering_after_fusion
            or self.node.get_device().type != "cuda"
        )

        if isinstance(self.node, ir.TemplateBuffer):
            self.set_read_writes(
                self.node.extract_read_writes(normalize=should_normalize)
            )
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=should_normalize
                )
            )

    def recompute_size_and_body(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._compute_attrs(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )

    def refresh_dependencies(self, normalize: bool) -> None:
        # Fake dependencies are added manually. They can not be analyzed from
        # extract_read_writes. Find them out and apply manually.
        fake_deps = {
            dep for dep in self.read_writes.reads if isinstance(dep, (WeakDep, StarDep))
        }

        # don't normalize since the loop order may need to be further changed
        # later
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=normalize
            ).with_read(fake_deps)
        )

    def apply_new_loop_order(self, new_order: Sequence[int]) -> None:
        self._body = self._body.reorder_iter_loops(
            new_order,
        )
        self._sizes = self._body.sizes

        self.refresh_dependencies(normalize=False)

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> None:
        new_order = None
        self_sizes = self._sizes[0]
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if new_order:
            metrics.num_loop_reordering += 1
            loop_ordering_log.debug(
                "Reorder loops for %s with order %s", self.get_name(), new_order
            )
            self.apply_new_loop_order(new_order)
        else:
            loop_ordering_log.debug(
                "Don't reordering %s because we can not decide the suitable loop order",
                self.get_name(),
            )

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            if not isinstance(dep, WeakDep):
                buf_name = dep.name
                buf = V.graph.get_buffer(buf_name)
                lines.append(f"{buf_name}_layout = {pformat(buf.layout)}")
        if isinstance(self._body, LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        assert self.node is not None
        lines.extend(self._debug_str_for_device())

        return "\n".join(lines)

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        return self._sizes

    def is_reduction(self) -> bool:
        assert isinstance(
            self.node, (ir.ComputedBuffer, ir.TemplateBuffer)
        ), f"{type(self.node)=}"
        return bool(self.node.get_reduction_type())

    def is_split_scan(self) -> bool:
        assert isinstance(
            self.node, (ir.ComputedBuffer, ir.TemplateBuffer)
        ), f"{type(self.node)=}"
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars: Sequence[sympy.Expr]) -> None:
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def ranges_from_index_vars(
        self, index_vars: Sequence[Sequence[sympy.Expr]]
    ) -> Dict[sympy.Expr, sympy.Expr]:
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> None:
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    @cache_on_self
    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes
        return dependencies.extract_read_writes(
            self._body, sizes, hidden_args=[[sympy.Integer(0)] * len(reduction_sizes)]
        )

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        if self.is_template():
            return False
        if any(out.get_aliases() for out in self.get_outputs()):
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            assert isinstance(write_dep, dependencies.MemoryDep), f"{type(write_dep)=}"
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet[str]:
        buffers_store_as_atomic_add: OrderedSet[str] = OrderedSet()
        if isinstance(self._body, LoopBody):
            for node in self._body.get_nodes():
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        return buffers_store_as_atomic_add


def refresh_group_node_dependencies(group_snode: BaseSchedulerNode) -> None:
    snodes = group_snode.snodes  # type: ignore[attr-defined]
    group_snode.set_read_writes(
        dependencies.ReadWrites.merge_list([x.read_writes for x in snodes])
    )

    group_snode.unmet_dependencies = (
        OrderedSet(
            dep
            for dep in OrderedSet.union(*[x.unmet_dependencies for x in snodes])
            if dep.name not in group_snode.get_buffer_names()
        )
        - group_snode.read_writes.writes
    )


def init_group_node(
    group_snode: BaseSchedulerNode,
    scheduler: Scheduler,
    snodes: List[BaseSchedulerNode],
) -> None:
    assert isinstance(group_snode, (FusedSchedulerNode, GroupedSchedulerNode))
    group_snode.snodes = snodes
    group_snode.scheduler = scheduler
    group_snode.node = None
    group_snode.ancestors = OrderedSet.union(
        *[x.ancestors for x in snodes if x.ancestors is not None]
    )

    refresh_group_node_dependencies(group_snode)

    group_snode.min_order = min(x.min_order for x in group_snode.snodes)
    group_snode.max_order = max(x.max_order for x in group_snode.snodes)
    group_snode.outputs_by_name = {
        buf.get_name(): buf for buf in group_snode.get_outputs()
    }


class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    snodes: List[BaseSchedulerNode]

    @classmethod
    def fuse(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        assert node1.scheduler is node2.scheduler
        assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
        assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> None:
        if self.is_template():
            # We can not really reorder loops for a triton template
            return
        self_sizes = None
        for snode in self.snodes:
            assert isinstance(snode, SchedulerNode)
            if self_sizes is not None and self_sizes != snode._sizes[0]:
                loop_ordering_log.debug(
                    "Can not reorder fused node due to different sizes"
                )
                return
            self_sizes = snode._sizes[0]
        new_order = None

        assert self_sizes is not None
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if not new_order:
            loop_ordering_log.debug(
                "Dont reordering fused node %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return
        metrics.num_loop_reordering += 1
        loop_ordering_log.debug(
            "Reorder loops for fused node %s with order %s", self.get_name(), new_order
        )
        for snode in self.snodes:
            assert isinstance(snode, SchedulerNode)
            snode.apply_new_loop_order(new_order)  # type: ignore[arg-type]

        refresh_group_node_dependencies(self)

    def __init__(self, scheduler: Scheduler, snodes: List[BaseSchedulerNode]) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        self.users: List[NodeUser] = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> List[SchedulerBuffer]:
        result: List[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    def debug_str_extra(self) -> str:
        lines = [
            f"{self.get_name()}.snodes[{i}] =\n{node.debug_str()}"
            for i, node in enumerate(self.snodes)
        ]
        node = self.snodes[0].node
        if node is not None:
            lines.extend(self._debug_str_for_device())

        return textwrap.indent("\n".join(lines).rstrip(), "    ")

    def debug_str_short(self) -> str:
        snodes_str = [node.debug_str_short() for node in self.snodes]
        return f"{self}, snodes: {snodes_str}"

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
    ) -> None:
        # Set self.last_usage using the global information
        # This will be used for inter-kernel optimisations
        super().set_last_usage(future_used_buffers, mutation_real_name)
        # Set self.last_usage on the snodes
        # This will be used for optimisations within the kernel
        future_used_buffers: OrderedSet[str] = OrderedSet()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(
            *[x.used_or_aliased_buffer_names() for x in self.snodes]
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return self.snodes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self) -> bool:
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_split_scan(self) -> bool:
        return any(x.is_split_scan() for x in self.snodes)

    @cache_on_self
    def is_template(self) -> bool:
        return any(x.is_template() for x in self.snodes)

    @cache_on_self
    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        for node in self.snodes:
            if node.is_template():
                return node.get_template_node()
        return None

    def get_device(self) -> torch.device:
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool:
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: Dict[str, str]) -> None:
        raise NotImplementedError

    def add_fake_dep(self, name: Dep) -> None:
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        raise NotImplementedError

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        node_typestr = ",".join(type(n).__name__ for n in self.snodes)
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({node_typestr})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.outputs = [
            """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()


class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """
    This is a schedular node that consists of a set of scheduler nodes that
    has no data dependencies among them and can be executed in parallel.
    """

    def get_consumer_subnode_for(
        self, producer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        for buf in producer.get_outputs():
            if buf.get_name() in self.read_to_node:
                return self.read_to_node[buf.get_name()]

        return None

    def get_producer_subnode_for(
        self, consumer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        producers = OrderedSet[BaseSchedulerNode]()
        for rd in consumer.read_writes.reads:
            if rd.name not in self.scheduler.name_to_buf:
                continue

            node_name = self.scheduler.name_to_buf[rd.name].defining_op.get_name()
            if node_name in self.name_to_node:
                producers.add(self.name_to_node[node_name])

        # Don't permit fusion if there are multiple subnodes
        # that this consumer reads from
        if len(producers) == 1:
            return next(iter(producers))
        else:
            return None

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        why = WhyNoFuse(producer, consumer)
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            foreach_match = len(producer.snodes) == len(consumer.snodes)
            if not foreach_match:
                why("foreach do not have same length")
            return foreach_match and all(
                producer.scheduler.can_fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():
            if producer.is_reduction():
                why(
                    "candidate producer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            why("candidate producer is not dep of any foreach consumer")
            return False

        elif producer.is_foreach():
            if consumer.is_reduction():
                why(
                    "candidate consumer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            why("candidate consumer has no dep in any foreach producer")
            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    def fuse(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> ForeachKernelSchedulerNode:
        assert producer.is_foreach() or consumer.is_foreach()
        if producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            use_custom_partition_algo = producer.use_custom_partition_algo
            enable_autotune = producer.enable_autotune
        else:
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            use_custom_partition_algo = consumer.use_custom_partition_algo
            enable_autotune = consumer.enable_autotune
        prev_node_1 = None
        prev_node_2 = None
        fused_nodes: List[BaseSchedulerNode]
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]
        elif producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        elif consumer.is_foreach():
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None

            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        else:
            raise AssertionError(
                "At least one node passed to ForeachKernelSchedulerNode.fuse should be a foreach node"
            )

        return cls(
            producer.scheduler,
            fused_nodes,
            use_custom_partition_algo=use_custom_partition_algo,
            prev_node_1=prev_node_1,
            prev_node_2=prev_node_2,
            enable_autotune=enable_autotune,
        )

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: List[BaseSchedulerNode],
        use_custom_partition_algo: bool,
        prev_node_1: Optional[BaseSchedulerNode] = None,
        prev_node_2: Optional[BaseSchedulerNode] = None,
        enable_autotune: bool = False,
    ) -> None:
        self.read_to_node = {}
        self.name_to_node = {}

        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, snodes)

            for node in snodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                for name in node.get_operation_names():
                    self.name_to_node[name] = node
        else:
            self.scheduler = scheduler
            self.snodes = snodes
            self.node = None
            self.users: List[NodeUser] = []

            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            self.unmet_dependencies = (
                OrderedSet(
                    dep
                    for dep in OrderedSet.union(
                        prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                    )
                    if dep.name not in self.get_buffer_names()
                )
                - self.read_writes.writes
            )

            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])

            if prev_node_1.is_foreach():
                assert isinstance(prev_node_1, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_1, prev_node_2
            else:
                assert isinstance(prev_node_2, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_2, prev_node_1

            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_operation_names():
                self.name_to_node[name] = other_node

        self.use_custom_partition_algo = use_custom_partition_algo
        self.group = (snodes[0].get_device(), ((sympy.Expr("combo_kernel"),),))
        self.origins: OrderedSet[torch.fx.Node] = OrderedSet()
        self.enable_autotune = enable_autotune

    @classmethod
    def combinable_nodes(
        cls, nodes: List[BaseSchedulerNode]
    ) -> List[BaseSchedulerNode]:
        extern = [x for x in nodes if isinstance(x, ExternKernelSchedulerNode)]
        if extern:
            log.debug(
                "ComboKernels: %d external nodes are filtered %s",
                len(extern),
                [node.node.get_origins() for node in extern if node.node is not None],
            )
        filtered_nodes = [
            x
            for x in nodes
            if not isinstance(x, (NopKernelSchedulerNode, ExternKernelSchedulerNode))
        ]
        foreach_nodes = [
            x for x in filtered_nodes if isinstance(x, ForeachKernelSchedulerNode)
        ]
        if foreach_nodes:
            log.debug("ComboKernels: %d foreach nodes are filtered", len(foreach_nodes))
        filtered_nodes = [
            x for x in filtered_nodes if not isinstance(x, ForeachKernelSchedulerNode)
        ]
        template_nodes = [x for x in filtered_nodes if x.is_template()]
        if template_nodes:
            log.debug(
                "ComboKernels: %d template nodes are filtered", {len(template_nodes)}
            )
        filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]
        return filtered_nodes

    @staticmethod
    def _default_group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> List[List[BaseSchedulerNode]]:
        """
        Returns a list of lists of nodes that are to be grouped together.
        """
        sorted_nodes = scheduler._topological_sort_nodes()
        grouped_nodes = []
        max_num_nodes = 8
        for nodes in sorted_nodes:
            grouped_nodes.extend(
                [
                    nodes[i : i + max_num_nodes]
                    for i in range(0, len(nodes), max_num_nodes)
                ]
            )

        return grouped_nodes

    group_algorithm_for_combo_kernels: Callable[
        [Scheduler], List[List[BaseSchedulerNode]]
    ] = _default_group_nodes_for_combo_kernels

    @staticmethod
    def set_group_algorithm_for_combo_kernels(
        custom_group_algorithm: Callable[[Scheduler], List[List[BaseSchedulerNode]]]
    ) -> None:
        ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels = (
            custom_group_algorithm
        )

    @staticmethod
    def group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> List[List[BaseSchedulerNode]]:
        return ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels(scheduler)

    def mark_run(self) -> None:
        raise NotImplementedError

    def codegen(self) -> None:
        assert isinstance(self.node, ir.ComputedBuffer), f"{type(self.node)=}"
        self.node.get_store_function()(self.node.make_loader()())

    def is_foreach(self) -> bool:
        return True

    def get_subkernel_nodes(self) -> List[BaseSchedulerNode]:
        """Returns a list of nodes which comprise the combo kernel.
        These nodes may be vertically fused."""
        return list(self.snodes)

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        """Returns all nodes contained in this kernel, unpacking fused nodes
        into their constituent scheduler nodes."""
        return list(itertools.chain.from_iterable(x.get_nodes() for x in self.snodes))

    def get_first_name(self) -> str:
        return self.snodes[0].get_first_name()

    def prune_redundant_deps(
        self, name_to_fused_node: Dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

        for node in self.snodes:
            node.prune_redundant_deps(name_to_fused_node)


class GroupedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be *grouped* together (it does not allow another node to be scheduled
    in between its constituent nodes, nor does it allow another node to fuse into any of its constituent nodes).
    The way it does this is by maintaining its unmet dependencies as the union of its constituent nodes.
    Fusion will still happen among the nodes within each GroupedSchedulerNode.
    At codegen time, this scheduler node will be unpacked and codegen is called on each constituent node.
    """

    snodes: List[BaseSchedulerNode]

    @classmethod
    def create(cls, snodes: List[BaseSchedulerNode]) -> GroupedSchedulerNode:
        scheduler = snodes[0].scheduler
        assert all(node.scheduler is scheduler for node in snodes)
        grouped_snode = cls(scheduler, snodes)  # type: ignore[arg-type]
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped_snode
        scheduler.name_to_fused_node[grouped_snode.get_name()] = grouped_snode
        return grouped_snode

    def __init__(self, scheduler: Scheduler, snodes: List[BaseSchedulerNode]) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)

    def unpack(self) -> List[BaseSchedulerNode]:
        """
        Do fusion among nodes within this GroupedSchedulerNode,
        and then unpack this GroupedSchedulerNode into regular nodes.
        """
        for snode in self.snodes:
            self.scheduler.name_to_fused_node[snode.get_name()] = snode
        del self.scheduler.name_to_fused_node[self.get_name()]
        return self.scheduler.fuse_nodes(self.snodes)

    def add_fake_dep(self, fake_dep: Dep) -> None:
        self.set_read_writes(self.read_writes.with_read(fake_dep))
        self.unmet_dependencies.add(fake_dep)

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> List[SchedulerBuffer]:
        result: List[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return self.snodes

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        # GroupedSchedulerNode cannot be fused with another node
        return False


def pick_loop_order(
    stride_lengths: List[List[int]],
    sizes: List[sympy.Expr],
    priority_idx: Tuple[int, ...] = (),
) -> List[int]:
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        # Take abs, otherwise flipped dimensions are treated as smaller
        # strides than contiguous dims
        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]

        # equivalent to
        # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
        a_first = sum(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = sum(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # if we have priority node, only use that node's order
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


@dataclasses.dataclass
class NodeUser:
    node: Union[BaseSchedulerNode, OutputNode]
    can_inplace: bool = False

    # A weak user must be scheduled after a given node, but doesn't actually
    # use the result
    is_weak: bool = False

    def __hash__(self) -> int:
        return hash((self.node.get_name(), self.can_inplace, self.is_weak))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NodeUser)
            and self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak
        )

    def get_name(self) -> str:
        return self.node.get_name()

    def merge(self, other: NodeUser) -> NodeUser:
        assert self.node is other.node
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


_post_grad_graph_counter = itertools.count()


class Scheduler:
    __dep_size_hint_cache: Dict[Dep, int]

    def __init__(self, nodes: List[ir.Operation]) -> None:
        with dynamo_timed("Scheduler.__init__"):
            self._init(nodes)

    def _init(self, nodes: List[ir.Operation]) -> None:
        super().__init__()
        self.__dep_size_hint_cache = {}
        V.graph.scheduler = self
        self.backends: Dict[torch.device, BaseScheduling] = {}
        self.post_grad_graph_id = next(_post_grad_graph_counter)

        self.completed_operations: OrderedSet[str] = OrderedSet()
        self.available_buffer_names = OrderedSet(
            [
                *V.graph.graph_inputs.keys(),
                *V.graph.constants.keys(),
                *V.graph.torchbind_constants.keys(),
            ]
        )

        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.update_zero_dim_cpu_tensor()
        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        self.name_to_node: Dict[str, BaseSchedulerNode] = {
            n.get_name(): n for n in self.nodes
        }
        self.name_to_buf: Dict[str, SchedulerBuffer] = {
            buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
        }
        self.name_to_fused_node: Dict[str, BaseSchedulerNode] = self.name_to_node.copy()

        # mutation_real_name: Maps back to the original name for codegen
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_real_name = {"buf0" : "buf1"}
        # all subsequent uses of buf0 become buf1's usage in dependency graph
        self.mutation_real_name: Dict[str, str] = {}

        # We handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_renames = {"buf1" : "buf0"}
        # in codegen we only use buf0, never buf1
        self.mutation_renames: Dict[str, str] = {}

        self.compute_dependencies()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.dead_node_elimination()
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.compute_ancestors()
        if config.reorder_for_compute_comm_overlap:
            self.nodes = comms.decide_global_ordering_of_comms(
                self.nodes,
                self.name_to_buf,
                self.name_to_fused_node,
            )

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        V.debug.ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.create_foreach_nodes()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.logged_slow_fusion: OrderedSet[Tuple[str, str]] = OrderedSet()
        if config._pre_fusion_custom_pass is not None:
            self.nodes = config._pre_fusion_custom_pass(self.nodes)
        self.nodes = self.fuse_nodes(self.nodes)
        if config.reorder_for_peak_memory:
            from .memory import reorder_for_peak_memory

            self.nodes = reorder_for_peak_memory(
                self.nodes,
                self.name_to_buf,
                self.name_to_fused_node,
                OrderedSet(V.graph.graph_inputs.keys()),
                OrderedSet(V.graph.get_output_names()),
            )
        self.merge_loops()
        self.finalize_multi_template_buffers()
        if config.reorder_for_compute_comm_overlap:
            self.nodes = comms.reorder_compute_and_comm_for_overlap(self.nodes)
        if config.combo_kernels:
            self.create_combo_kernel_nodes(num_ck_nodes=None)
        self.process_grouped_nodes()
        self.compute_last_usage()
        V.debug.ir_post_fusion(self.nodes)
        V.debug.graph_diagram(self.nodes)
        self.debug_draw_graph()

        # used during codegen:
        self.buffer_names_to_free: OrderedSet[str] = OrderedSet()

        # fx graph node to the position it appears in the graph
        # for debug attribution
        self.origin_to_index: Dict[torch.fx.Node, int] = {}

        get_metric_table("graph_stats").add_row(
            lambda: {
                "graph_id": self.post_grad_graph_id,
                "num_nodes_before_fusion": self.num_orig_nodes,
                "num_nodes_after_fusion": len(self.nodes),
            }
        )

    @property
    def current_device(self) -> Optional[torch.device]:
        return V.graph.current_device

    @current_device.setter
    def current_device(self, device: Optional[torch.device]) -> None:
        V.graph.current_device = device

    def debug_draw_graph(self) -> None:
        """Generate an image of the graph for debugging"""
        if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
            from .debug import draw_buffers

            draw_buffers(self.nodes, print_graph=True)

    def debug_print_nodes(self, label: str) -> None:
        if log.isEnabledFor(logging.INFO):
            log.info("%s:", label)
            for node in self.nodes:
                node.log_details()

    def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
        assert (
            node.get_origins() is not None
        ), "All nodes passed to scheduling must have an origin"
        if node.is_no_op():
            return NopKernelSchedulerNode(self, node)
        elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            return SchedulerNode(self, node)
        elif isinstance(node, ir.ExternKernel):
            return ExternKernelSchedulerNode(self, node)
        else:
            raise NotImplementedError(node)

    def create_foreach_nodes(self) -> None:
        removed_node_names: OrderedSet[str] = OrderedSet()
        fe_nodes = []
        kept_node_names = self.name_to_fused_node.keys()

        for names in V.graph.lists.values():
            names = [
                name
                for name in names
                if name in kept_node_names
                and not isinstance(self.name_to_node[name], NopKernelSchedulerNode)
            ]
            if not names:
                # All nodes eliminated
                continue

            removed_node_names.update(names)
            snodes = [self.name_to_node[name] for name in names]

            enable_autotune = config.combo_kernels_autotune > 1
            fe_node = ForeachKernelSchedulerNode(
                self,
                snodes,
                use_custom_partition_algo=False,
                enable_autotune=enable_autotune,
            )

            fe_nodes.append(fe_node)

            for name in names:
                self.name_to_fused_node[name] = fe_node

        self.nodes = [
            node for node in self.nodes if node.get_name() not in removed_node_names
        ] + list(fe_nodes)

    def compute_dependencies(self) -> None:
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """

        T = TypeVar("T")

        class DedupList(Generic[T]):
            """
            This data structure behaves like a list except it makes sure the
            elements remain unique.
            Normally one could use a OrderedSet/dict for this purpose however
            the list in question gets elements appended as it is being
            iterated over which means that we need to keep the list
            semantics.
            """

            def __init__(
                self,
                items: Optional[List[T]] = None,
                membership: Optional[OrderedSet[T]] = None,
            ) -> None:
                self.items = items or []
                self.membership = membership or OrderedSet()

            def append(self, node_user: T) -> None:
                if node_user in self.membership:
                    return
                self.items.append(node_user)
                self.membership.add(node_user)

            def __add__(self, other: DedupList[T]) -> DedupList[T]:
                new_membership = OrderedSet.union(self.membership, other.membership)
                new_items = self.items + [
                    x for x in other.items if x not in self.membership
                ]
                return DedupList(new_items, new_membership)

        name_to_users: DefaultDict[str, DedupList[NodeUser]] = collections.defaultdict(
            DedupList
        )

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node in self.nodes:
            for buf1 in node.get_outputs():
                buf1_name = buf1.get_name()
                for buf2_name in buf1.get_aliases():
                    if buf1_name in name_to_users and buf2_name in name_to_users:
                        # merge the two
                        list1 = name_to_users[buf1_name]
                        list2 = name_to_users[buf2_name]
                        combined = list1 + list2
                        for key in name_to_users.keys():
                            if (
                                name_to_users[key] is list1
                                or name_to_users[key] is list2
                            ):
                                name_to_users[key] = combined
                    elif buf1_name in name_to_users:
                        name_to_users[buf2_name] = name_to_users[buf1_name]
                    else:
                        name_to_users[buf1_name] = name_to_users[buf2_name]

        def rename(n: str) -> str:
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def add_user(
            used_by_name: str,
            user_node: Union[BaseSchedulerNode, OutputNode],
            can_inplace: bool = False,
            is_weak: bool = False,
        ) -> None:
            name_to_users[rename(used_by_name)].append(
                NodeUser(user_node, can_inplace, is_weak)
            )

        unbacked_symbol_to_origin_node: Dict[sympy.Symbol, Optional[str]] = {}

        # NB: None means that the dependency is on an input.  Don't actually
        # generate a dependency because if we do, Inductor will start trying
        # to free the unbacked int but that's pointless
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                for fs in val.free_symbols:
                    unbacked_symbol_to_origin_node[fs] = None

        for node in self.nodes:
            log.debug("scheduling %s", node.node)

            # unbacked symbols don't follow ordinary buffer dependencies, so
            # we track their def/uses separately
            assert node.node is not None
            unbacked_symbol_defs = sorted(
                node.node.get_unbacked_symbol_defs(), key=lambda x: x.name
            )
            for s in unbacked_symbol_defs:
                assert isinstance(s, sympy.Symbol)
                # Pick the first definer as canonical.  There may be multiple
                # because if a MultiOutputLayout buffer propagates an unbacked
                # symint to multiple outputs, they will all claim to def it.
                if s not in unbacked_symbol_to_origin_node:
                    unbacked_symbol_to_origin_node[s] = node.get_name()

            unbacked_symbol_uses = sorted(
                node.node.get_unbacked_symbol_uses(), key=lambda x: x.name
            )
            # if a kernel takes unbacked symints, register dependencies
            for s in unbacked_symbol_uses:
                assert (
                    s in unbacked_symbol_to_origin_node
                ), f"{s} not in {unbacked_symbol_to_origin_node}"
                if (r := unbacked_symbol_to_origin_node[s]) is not None:
                    for buf in self.name_to_node[r].get_outputs():
                        node.add_fake_dep(StarDep(buf.get_name()))

            if (
                len(node.read_writes.writes) == 1
                and (dep := next(iter(node.read_writes.writes)))
                and isinstance(dep, MemoryDep)
            ):
                node_mode = dep.mode
            else:
                node_mode = None

            # Handle output mutations
            for buf in node.get_outputs():
                # a node will mutate either 0 or 1 buffers
                assert len(buf.get_mutations()) <= 1
                for alt_name in buf.get_mutations():
                    alt_name = rename(alt_name)
                    # this node must run after the prior writer
                    add_user(alt_name, node)
                    node.add_fake_dep(StarDep(alt_name, mode=node_mode))
                    for user in name_to_users[alt_name].items:
                        if user.get_name() == node.get_name():
                            continue

                        assert isinstance(user.node, BaseSchedulerNode)
                        for other_name in user.node.get_buffer_names():
                            # this node must run after all prior readers
                            other_name = rename(other_name)
                            node.add_fake_dep(
                                WeakDep(other_name, mutating_buf=buf.get_name())
                            )
                            add_user(other_name, node, is_weak=True)

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                if not isinstance(read, WeakDep):
                    add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for buf in node.get_outputs():
                for alt_name in buf.get_mutations():
                    self.mutation_renames[rename(alt_name)] = buf.get_name()
                    self.mutation_renames[alt_name] = buf.get_name()
                    self.mutation_real_name[
                        buf.get_name()
                    ] = self.mutation_real_name.get(alt_name, alt_name)

        # make sure outputs aren't dead-code-eliminated
        for buf_name in V.graph.get_output_names():
            log.debug("scheduling output %s", buf_name)
            add_user(buf_name, OutputNode(StarDep(buf_name)))

        # make sure unbacked symints aren't dead-code-eliminated
        for out in V.graph.graph_outputs:
            for s in out.get_unbacked_symbol_uses():
                assert (
                    s in unbacked_symbol_to_origin_node
                ), f"{s} not in {unbacked_symbol_to_origin_node.keys()}"
                if r := unbacked_symbol_to_origin_node[s]:
                    for buf_name in self.name_to_node[r].get_buffer_names():
                        log.debug(
                            "scheduling output %s for unbacked symint %s", buf_name, s
                        )
                        add_user(buf_name, OutputNode(StarDep(buf_name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)
            elif name in V.graph.constants:
                # In AOTI, module parameters and buffers are not lifted as graph inputs
                add_user(name, OutputNode(StarDep(name)))

        inp_names = {
            name: index for index, name in enumerate(V.graph.graph_inputs.keys())
        }
        V.graph.mutated_input_idxs = [
            inp_names[name] for name in V.graph.mutated_inputs
        ]

        # copy users information onto the nodes
        for node in self.nodes:
            for buf in node.get_outputs():
                buf.set_users(name_to_users[buf.get_name()].items)

    def dead_node_elimination(self) -> None:
        """
        Remove any nodes without users
        """
        # self.nodes is in topological order, so by iterating in reverse order
        # we have visited (and potentially removed) all users before visiting a
        # given node.
        updated_nodes = []
        for node in reversed(self.nodes):

            def can_eliminate_user(user: NodeUser) -> bool:
                return user.is_weak or user.get_name() in V.graph.removed_operations

            active_buffers = False
            for buf in node.get_outputs():
                can_eliminate = all(can_eliminate_user(u) for u in buf.users)
                if can_eliminate:
                    log.debug("removed dead buffer: %s", buf.get_name())
                    V.graph.removed_buffers.add(buf.get_name())
                else:
                    active_buffers = True

            can_eliminate = not node.has_side_effects() and not active_buffers

            if not can_eliminate:
                updated_nodes.append(node)
            else:
                # dead code
                log.debug("removed dead operation: %s", node.get_name())
                V.graph.removed_operations.add(node.get_name())

        self.nodes = list(reversed(updated_nodes))

        # Prune any WeakDeps no longer needed
        for node in self.nodes:
            node.prune_weak_deps()

    def topological_sort_schedule(
        self, nodes: List[BaseSchedulerNode]
    ) -> List[BaseSchedulerNode]:
        """
        Ensure nodes is in topologically sorted order
        """
        seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
        name_to_node: Dict[str, BaseSchedulerNode] = dict()
        result: List[BaseSchedulerNode] = []

        def visit(n: BaseSchedulerNode) -> None:
            if n not in seen:
                seen.add(n)
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    # We only care about doing toposort within `nodes`
                    if dep.name not in name_to_node:
                        continue
                    visit(name_to_node[dep.name])
                result.append(n)

        for node in nodes:
            for name in node.get_buffer_names():
                name_to_node[name] = node
        for node in nodes:
            visit(node)
        return result

    def _get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> List[BaseSchedulerNode]:
        unmet_deps = OrderedSet[str]()
        if isinstance(
            snode,
            (
                SchedulerNode,
                ExternKernelSchedulerNode,
                NopKernelSchedulerNode,
                FusedSchedulerNode,
            ),
        ):
            for dep in snode.unmet_dependencies:
                unmet_deps.add(dep.name)
        else:
            raise RuntimeError(
                f"get_unmet_dep_nodes is not implemented for {type(snode)}."
            )
        unmet_dep_ops = (self.name_to_buf[dep].defining_op for dep in unmet_deps)
        return list({self.name_to_fused_node[n.get_name()] for n in unmet_dep_ops})

    def _topological_sort_nodes(self) -> List[List[BaseSchedulerNode]]:
        """
        Sort nodes by their topological order, return a list of node lists.
        """
        order = []
        nodes = dict.fromkeys(self.nodes, 0)
        children: Dict[Any, Any] = {}
        for node in self.nodes:
            deps = self._get_unmet_dep_nodes(node)
            nodes[node] = len(deps)
            for dep in deps:
                c = children.get(dep, [])
                c.append(node)
                children[dep] = c

        zero_deg_nodes = [n for n, v in nodes.items() if v == 0]
        while zero_deg_nodes:
            order.append(zero_deg_nodes)
            for n in zero_deg_nodes:
                for user in children.get(n, []):
                    nodes[user] -= 1
                nodes.pop(n)
            zero_deg_nodes = [n for n, v in nodes.items() if v == 0]
        assert not nodes, "Topological sort failed!"
        return order

    def compute_ancestors(self) -> None:
        """
        Populate each node.ancestors
        """
        # note self.nodes is topologically sorted
        name_to_ancestors: Dict[str, OrderedSet[str]] = {}
        for node in self.nodes:
            ancestors: OrderedSet[str] = OrderedSet()
            for dep in node.unmet_dependencies:
                dep_node_name = self.name_to_buf[dep.name].defining_op.get_name()
                ancestors.add(dep_node_name)
                ancestors |= name_to_ancestors[dep_node_name]
            name_to_ancestors[node.get_name()] = ancestors
            node.ancestors = ancestors

        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order

    def merge_loops(self) -> None:
        for node in self.nodes:
            if not config.loop_ordering_after_fusion:
                continue

            # Even for CPU, if we are using the halide backend, we still need
            # the merge loops steps below
            if not isinstance(node, (SchedulerNode, FusedSchedulerNode)) or (
                node.get_device().type != "cuda" and config.cpu_backend != "halide"
            ):
                continue
            for snode in node.get_nodes():
                # merge loops for the scheduler node
                if not isinstance(snode, SchedulerNode) or snode.is_template():
                    continue

                snode._body = snode._body.merge_loops()
                snode._sizes = snode._body.sizes

                # merge_loops is called after loop reordering.
                # We still need retain fake dependencies since codegen the
                # estimated amount of memory access rely on them.
                snode.refresh_dependencies(normalize=True)

                # Note that for CPU backend, merging loops will change
                # snode.group. It's fine for Triton backend.
                # But if we simplify update snode.group like this:
                #   group_fn = self.get_backend(snode.node.get_device()).group_fn
                #   snode.group = (snode.node.get_device(), group_fn(snode._sizes))
                # There is still an issue due to different snode in a
                # FusedSchedulerNode having different merged loops.
                # Skip CPU backend for now.

    def fuse_nodes(self, nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.
        """
        for i in range(10):
            old_len = len(nodes)
            fusion_log.debug(
                "===== attempting fusion (%d/10): %d nodes =====",
                i + 1,
                old_len,
            )
            nodes = self.fuse_nodes_once(nodes)
            new_len = len(nodes)
            fusion_log.debug(
                "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
                i + 1,
                old_len,
                new_len,
            )
            if new_len == old_len or new_len == 1:
                fusion_log.debug("===== fusion complete (%d iterations) =====", i + 1)
                break
        return nodes

    def process_grouped_nodes(self) -> None:
        """
        Unpack GroupedSchedulerNode into regular nodes.
        """
        new_nodes: List[BaseSchedulerNode] = []
        for node in self.nodes:
            new_nodes.extend(
                node.unpack() if isinstance(node, GroupedSchedulerNode) else [node]
            )
        self.nodes = new_nodes

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        assert len(nodes) > 0
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        return backend.benchmark_fused_nodes(nodes)

    def finalize_multi_template_buffers(self) -> None:
        def replace_operation_buffer(
            orig_node: ir.MultiTemplateBuffer, new_node: ir.OperationBuffer
        ) -> None:
            replaced_buf_name = new_node.get_name()
            orig_buf_name = orig_node.get_name()
            assert isinstance(orig_buf_name, str) and isinstance(replaced_buf_name, str)

            replaced_op_name = new_node.get_operation_name()
            orig_op_name = orig_node.get_operation_name()
            assert isinstance(orig_op_name, str) and isinstance(replaced_op_name, str)

            del V.graph.name_to_buffer[replaced_buf_name]
            new_node.name = orig_buf_name

            del V.graph.name_to_op[replaced_op_name]
            new_node.operation_name = orig_op_name

            orig = V.graph.buffers.index(orig_node)
            V.graph.buffers.remove(new_node)
            V.graph.buffers[orig] = new_node
            V.graph.name_to_buffer[orig_buf_name] = new_node

            orig = V.graph.operations.index(orig_node)
            V.graph.operations.remove(new_node)
            V.graph.operations[orig] = new_node
            V.graph.name_to_op[orig_op_name] = new_node

        for i, node in enumerate(self.nodes):
            if isinstance(node, SchedulerNode) and isinstance(
                node.node, ir.MultiTemplateBuffer
            ):
                multi_node = node.node
                if not config.test_configs.force_extern_kernel_in_multi_template:
                    min_node_unfused, _ = multi_node.get_min_choice()
                else:
                    min_node_unfused = next(
                        (
                            timing
                            for timing in multi_node.choice_timings
                            if isinstance(
                                timing,
                                torch._inductor.select_algorithm.ExternKernelCaller,
                            )
                        ),
                        None,  # type: ignore[arg-type]
                    )
                    assert min_node_unfused is not None

                if isinstance(
                    min_node_unfused,
                    torch._inductor.ir.TritonTemplateCallerBase,
                ):
                    node.node.finalize_as_triton_caller(min_node_unfused)
                    continue

                out_tensorbox = min_node_unfused.output_node()
                out_storage = out_tensorbox.data
                assert isinstance(out_storage, ir.StorageBox)
                out_buffer = out_storage.data
                assert isinstance(out_buffer, ir.OperationBuffer)

                out_buffer.layout = multi_node.layout
                replace_operation_buffer(multi_node, out_buffer)
                new_scheduler_node = self.create_scheduler_node(out_buffer)

                self.nodes[i] = new_scheduler_node
                self.name_to_node[node.get_name()] = new_scheduler_node
                self.name_to_fused_node[node.get_name()] = new_scheduler_node

                for new_out, old_out in zip(
                    new_scheduler_node.get_outputs(), node.get_outputs()
                ):
                    self.name_to_buf[old_out.get_name()] = new_out
                    new_out.users = old_out.users

                new_scheduler_node.min_order = node.min_order
                new_scheduler_node.max_order = node.max_order
                new_scheduler_node.last_usage = node.last_usage

    def _any_atomic_add(self, node_list: Sequence[BaseSchedulerNode]) -> bool:
        return any(
            hasattr(n.node, "data")
            and n.node is not None
            and hasattr(n.node.data, "scatter_mode")
            and n.node.data.scatter_mode == "atomic_add"
            for n in node_list
        )

    def speedup_by_fusion(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """

        is_multi_template = node1.is_template() and isinstance(
            node1.get_template_node(), ir.MultiTemplateBuffer
        )
        if not config.benchmark_fusion and not is_multi_template:
            return True

        if (
            node1.is_template()
            and not isinstance(node1.get_template_node(), ir.TritonTemplateBuffer)
            or node1.is_foreach()
            or node2.is_foreach()
        ):
            # TODO support benchmarking epilogue fusion
            return True

        node_list_1 = node1.get_nodes()
        device = node_list_1[0].get_device()

        # don't support benchmark fusion for CPU right now.
        if device.type == "cpu":
            return True

        node_list_2 = node2.get_nodes()
        node_list_fused = list(itertools.chain(node_list_1, node_list_2))

        # We can not accurately benchmark kernel using atomic_add
        # due to how we generate random integer inputs.
        # Skip benchmarking them by allowing fusion.
        if self._any_atomic_add(node_list_fused):
            return True

        from triton.compiler.errors import CompilationError

        why = WhyNoFuse(node1, node2)

        def log_fusion(ms_fused: float, ms1: float, ms2: float) -> None:
            if fusion_log.isEnabledFor(logging.DEBUG):
                if ms_fused < ms1 + ms2:
                    fusion_log.debug(
                        "can fuse (benchmark): fusing %s with %s cause %sx speedup",
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        green_text(f"{(ms1 + ms2) / ms_fused:.3f}"),
                    )
                else:
                    fusion_log.debug(
                        "cannot fuse (benchmark): fusing %s with %s cause %sx slowdown",
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        red_text(f"{ms_fused / (ms1 + ms2):.3f}"),
                    )

        if isinstance(node1, SchedulerNode) and isinstance(
            node1.node, ir.MultiTemplateBuffer
        ):
            multi_node = node1.node
            choice_timings = multi_node.choice_timings

            _, ms1 = multi_node.get_min_choice()
            ms2, path2 = self.benchmark_fused_nodes(node_list_2)

            min_ms_fused = float("inf")
            ms_fused_choice = None

            triton_choices = 0

            for choice, unfused_time in sorted(
                choice_timings.items(), key=lambda x: x[1]
            ):
                if not isinstance(choice, torch._inductor.ir.TritonTemplateCallerBase):
                    continue

                if unfused_time >= ms1 + ms2:
                    break

                triton_choices += 1
                if triton_choices > config.max_epilogue_benchmarked_choices:
                    break

                # TODO - parallel compile triton templates
                # TODO - should prune/skip choices that are not within certain % of best choice
                with node1.node.swap_as_triton_caller(choice):
                    ms_fused, _ = self.benchmark_fused_nodes(node_list_fused)

                    if ms_fused < min_ms_fused:
                        min_ms_fused = ms_fused
                        ms_fused_choice = choice

            log_fusion(min_ms_fused, ms1, ms2)

            # after we do a fusion, we finalize a triton template.
            # TODO - could preserve multi template and choices for subsequent fusions
            if min_ms_fused < (ms1 + ms2) and ms_fused_choice is not None:
                node1.node.finalize_as_triton_caller(ms_fused_choice)
                return True
            else:
                return False
        else:
            try:
                ms1, path1 = self.benchmark_fused_nodes(node_list_1)
                if math.isinf(ms1):
                    why("register spilling of the first kernel")
                    return False
                ms2, path2 = self.benchmark_fused_nodes(node_list_2)
                if math.isinf(ms2):
                    why("register spilling of the second kernel")
                    return False
                ms_fused, path_fused = self.benchmark_fused_nodes(node_list_fused)
                if math.isinf(ms_fused):
                    why("register spilling of the fused kernel")
                    return False
            except CompilationError as e:
                # workaround triton issue: https://github.com/openai/triton/issues/2151
                if "Loop-carried variable" in str(e):
                    return True  # allow fusion
                else:
                    raise

        log_fusion(ms_fused, ms1, ms2)
        if (
            is_metric_table_enabled("slow_fusion")
            and ms_fused >= ms1 + ms2
            and (path1, path2) not in self.logged_slow_fusion
        ):
            self.logged_slow_fusion.add((path1, path2))
            get_metric_table("slow_fusion").add_row(
                lambda: {
                    "kernel1_path": path1,
                    "kernel1_latency": ms1,
                    "kernel2_path": path2,
                    "kernel2_latency": ms2,
                    "fused_kernel_path": path_fused,
                    "fused_kernel_latency": ms_fused,
                    "slow_down_ratio": ms_fused / (ms1 + ms2),
                }
            )
        return ms_fused < ms1 + ms2

    def fuse_nodes_once(
        self, nodes: List[BaseSchedulerNode]
    ) -> List[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuse(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = OrderedSet(nodes)
        if fusion_log.isEnabledFor(logging.DEBUG):
            fusion_log.debug("fuse_nodes_once, candidates:")
            for node in fused_nodes:
                fusion_log.debug("  " + node.debug_str_short())  # noqa: G003
        for node1, node2 in self.get_possible_fusions(nodes):
            node1 = self.name_to_fused_node[node1.get_first_name()]
            node2 = self.name_to_fused_node[node2.get_first_name()]
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                if not self.speedup_by_fusion(node1, node2):
                    continue
                fusion_log.debug(
                    "fusing %s with %s", node1.get_name(), node2.get_name()
                )

                # above can_fuse asserts that node2 has the same device
                device = node1.get_device()
                node3 = self.get_backend(device).fuse(node1, node2)
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        nodes = self.topological_sort_schedule(nodes)
        self.prune_redundant_deps(nodes)
        return nodes

    def create_combo_kernel_nodes(self, num_ck_nodes: Optional[int] = None) -> None:
        """
        Groups parallel nodes
        """
        fused_nodes = OrderedSet(self.nodes)
        count = 0
        num_nodes_orig = len(self.nodes)
        log.debug("ComboKernels: Generating with num_ck_nodes = %d...", num_ck_nodes)
        for num, node_list in enumerate(
            ForeachKernelSchedulerNode.group_nodes_for_combo_kernels(self)
        ):
            node_list = ForeachKernelSchedulerNode.combinable_nodes(node_list)
            if len(node_list) < 2:
                continue
            if num_ck_nodes is not None and count > num_ck_nodes:
                break
            if not self.speedup_by_combo_kernel(node_list):
                log.debug("ComboKernels: Not speeding up %d-th group", num)
                continue
            count += 1
            enable_autotune = config.combo_kernels_autotune > 0
            group_snode = ForeachKernelSchedulerNode(
                node_list[0].scheduler,
                node_list,
                use_custom_partition_algo=True,
                enable_autotune=enable_autotune,
            )
            log.info(
                "ComboKernels: Combining %d nodes for %d-th group",
                len(node_list),
                num,
            )
            for node in node_list:
                fused_nodes.remove(node)
            fused_nodes.add(group_snode)
            self.name_to_fused_node.update(
                {n.get_name(): group_snode for n in group_snode.get_nodes()}
            )
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.nodes = self.topological_sort_schedule(self.nodes)
        log.info(
            "Generated ComboKernel nodes: %d ComboKernels, totally %d -> %d nodels",
            count,
            num_nodes_orig,
            len(self.nodes),
        )
        self.prune_redundant_deps(self.nodes)

    def prune_redundant_deps(self, nodes: List[BaseSchedulerNode]) -> None:
        for node in nodes:
            node.prune_redundant_deps(self.name_to_fused_node)

    def get_possible_fusions(
        self, nodes: List[BaseSchedulerNode]
    ) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen: OrderedSet[Tuple[BaseSchedulerNode, BaseSchedulerNode]] = OrderedSet()

        def check_all_pairs(nodes: List[BaseSchedulerNode]) -> None:
            for node1_index, node1 in enumerate(nodes):
                for node2 in nodes[node1_index + 1 :]:
                    key = (node1, node2)
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.can_fuse(node1, node2):
                        possible_fusions.append(key)
                    elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(
                        node2, node1
                    ):
                        # foreach fusions and epilogue fusions are order dependent
                        possible_fusions.append((node2, node1))

        buffer_names_grouping = collections.defaultdict(list)
        for node in nodes:
            if self.unfusable_node(node):
                continue
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.aggressive_fusion:
            group_grouping = collections.defaultdict(list)
            for node in nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        possible_fusions = self.get_possible_fusions_with_highest_priority(
            possible_fusions
        )
        possible_fusions.sort(key=self.score_fusion_key, reverse=True)
        fusion_log.debug("found %d possible fusions", len(possible_fusions))
        return possible_fusions

    def will_fusion_create_cycle(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Finds whether there's a path from node1 to node2 (or vice-versa)
        caused indirectly by other fusions.
        """
        # since we are just returning boolean here, use slightly faster, unordered set
        visited: OrderedSet[FusedSchedulerNode] = OrderedSet()

        def found_path(node: BaseSchedulerNode) -> bool:
            # only fused nodes can introduce new ancestors.
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                if node.get_operation_names().issubset(combined_ancestors):
                    # All fusion outputs are in ancestors of node1 and node2, thus
                    # cannot introduce new path:
                    #
                    # 1. if output is neither descendent of node1 or node2, the
                    #        output cannot introduce a path
                    # 2. due to [can_fuse]: if WLOG output is descendent of node1, it cannot be
                    #        on path(node1->node2), hence it cannot be ancestor of node2
                    # 3. due to [acyclic]: if WLOG output is descendent of node1, it cannot be
                    #        ancestor of node1
                    return False
                else:
                    # continue DFS of new ancestors introduced by the fusion
                    return bool(combined_names & node.ancestors) or any(
                        found_path(self.name_to_fused_node[n])
                        for n in node.ancestors - combined_ancestors
                    )
            return False

        # as above - use slightly faster, unordered set
        combined_names = (
            node1.get_operation_names()._dict.keys()
            | node2.get_operation_names()._dict.keys()
        )
        combined_ancestors = (
            node1.ancestors._dict.keys() | node2.ancestors._dict.keys()
        ) - combined_names
        cycle = any(found_path(self.name_to_fused_node[n]) for n in combined_ancestors)
        if cycle:
            WhyNoFuse(node1, node2)("will create cycle")
        return cycle

    def can_fusion_increase_peak_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heurisitic would be to use live
        intervals of the buffers, find region of peak pressure in the original
        program and prevent fusion that crosses that peak region. We might need
        special care or good approximation in this implementation, as fusion of
        node changes live intervals, and re-computing live intervals and peak
        memory after each fusion can introduce large compilation overhead.
        """
        proximity_score = max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return proximity_score > 64

    def decide_fusion_fail_reason(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        common_buf_names: Tuple[str, ...],
    ) -> str:
        """
        Try to decide reasons why fusion fail due to no shared memory even though
        there are common buffers.
        """
        reasons = {}
        node1_name2dep = {dep.name: dep for dep in node1.read_writes.reads_and_writes()}
        node2_name2dep = {dep.name: dep for dep in node2.read_writes.reads_and_writes()}

        for buf_name in common_buf_names:
            buf = V.graph.get_buffer(buf_name)
            lhs_dep = node1_name2dep[buf_name]
            rhs_dep = node2_name2dep[buf_name]

            if lhs_dep.get_numel() != rhs_dep.get_numel():
                reasons[
                    buf_name
                ] = f"different numel: {lhs_dep.get_numel()} v.s. {rhs_dep.get_numel()}"
                continue

            # same numel but different MemoryDep.size. Should be broadcasting
            if sympy_product(lhs_dep.size) != sympy_product(rhs_dep.size):
                reasons[buf_name] = "broadcast"
                continue

            if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
                reasons[
                    buf_name
                ] = f"not MemoryDep: {type(lhs_dep)} v.s. {type(rhs_dep)}"
                continue

            lhs_off = lhs_dep.get_offset()
            rhs_off = rhs_dep.get_offset()
            if lhs_off != rhs_off:
                # One example is in transformer, we use a concatenated linear layer
                # to project Q/K/V and then split the result. The 3 splits will
                # point to the same buffer with different offsets.
                reasons[buf_name] = f"different offset: {lhs_off} v.s. {rhs_off}"
                continue

            if (
                lhs_dep.normalize_with_stride_order()
                == rhs_dep.normalize_with_stride_order()
            ):
                reasons[buf_name] = f"Mismatch loop orders: {lhs_dep} v.s. {rhs_dep}"
                continue

            # Add more rules here
            reasons[
                buf_name
            ] = f"Unknown reason: {lhs_dep} v.s. {rhs_dep}. Layout: {buf.layout}"

        return str(reasons)

    def has_shared_data_after_reordering_loop(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Right now just greedily reorder the loop of node1 to be compatible with node2,
        but ideally we should have some heuristics to reorder the loop for node2
        to be compatibile with node1 if that's more efficient.
        """

        # TODO Don't do loop reordering for CPU for now.
        # Should debug more why it does not work for CPU codegen
        if not config.loop_ordering_after_fusion or any(
            n.get_device().type == "cpu" for n in [node1, node2]
        ):
            return False

        node1_buffer_names = node1.read_writes.buffer_names()
        node2_buffer_names = node2.read_writes.buffer_names()
        # Fast path: no common buffers.
        common_buffer_names = node1_buffer_names & node2_buffer_names
        if not common_buffer_names:
            return False

        node1_name2dep = {dep.name: dep for dep in node1.read_writes.reads_and_writes()}
        node2_name2dep = {dep.name: dep for dep in node2.read_writes.reads_and_writes()}

        # Find the commons buffers that has different loop orders
        candidates = []
        for buffer_name in common_buffer_names:
            lhs_dep = node1_name2dep[buffer_name]
            rhs_dep = node2_name2dep[buffer_name]
            if (
                lhs_dep.normalize_with_stride_order()
                == rhs_dep.normalize_with_stride_order()
            ):
                candidates.append(
                    (
                        V.graph.sizevars.size_hint(lhs_dep.get_numel(), fallback=0),
                        lhs_dep,
                        rhs_dep,
                    )
                )

        if len(candidates) == 0:
            return False

        # Pick the largest buffer to guide the loop reordering
        numel, lhs_dep, rhs_dep = max(candidates, key=lambda x: x[0])

        if lhs_dep.num_vars != rhs_dep.num_vars:
            # this can happen due to we don't merge loops.
            # We can not do loop reordering in this case right now
            # Simply returning true if the two Deps are the same after
            # normalization (merging loops)
            return lhs_dep.normalize() == rhs_dep.normalize()

        # Only reorder loops for pointwise for now
        if not node1.is_reduction():
            node1.reorder_loops_by_dep_pair(lhs_dep, rhs_dep)
        elif not node2.is_reduction():
            node2.reorder_loops_by_dep_pair(rhs_dep, lhs_dep)
        else:
            loop_ordering_log.debug(
                "Don't reorder loops since both nodes are reductions: %s v.s. %s",
                node1.get_name(),
                node2.get_name(),
            )

        return (
            self.score_fusion_memory(node1, node2)
            >= config.score_fusion_memory_threshold
        )

    def unfusable_node(self, node: BaseSchedulerNode) -> bool:
        """
        Is this node unfusable under any conditions.
        """
        return (
            isinstance(node, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node.is_template()
        )

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """

        if node1 is node2:
            return False

        why = WhyNoFuse(node1, node2)

        if isinstance(node1, GroupedSchedulerNode) or isinstance(
            node2, GroupedSchedulerNode
        ):
            why("grouped node must not be fused with other nodes")
            return False
        if (
            isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node1.is_template()
        ):
            why("node1 is extern or nop")
            return False
        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            why("node2 is extern or nop")
            return False

        if node2.get_operation_names() & node1.ancestors:
            why("node1 must go before node2")
            return False

        if node2.is_template():
            why("templates can only fuse epilogues")
            return False
        if node1.is_template() and (
            node2.has_aliasing_or_mutation()
            or node2.is_reduction()
            or not config.epilogue_fusion
        ):
            why("template epilogue not satisfied")
            return False

        if (
            node1.get_buffer_names() | node2.get_buffer_names()
        ) & V.graph.no_fuse_buffer_names:
            why("fusion for buffer explicit disabled")
            return False

        device = node1.get_device()
        device2 = node2.get_device()
        if device != device2:
            why("device mismatch (%s vs %s)", device, device2)
            return False
        del device2

        no_shared_data = (
            self.score_fusion_memory(node1, node2)
            < config.score_fusion_memory_threshold
        )
        if no_shared_data:
            no_shared_data = not self.has_shared_data_after_reordering_loop(
                node1, node2
            )

        loop_ordering_log.debug(
            "%s and %s has%s shared data",
            node1.get_name(),
            node2.get_name(),
            " no" if no_shared_data else "",
        )
        if no_shared_data and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            if is_metric_table_enabled("fusion_failure_due_to_indexing_mismatch"):
                common_buf_names = (
                    node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
                )
                if len(common_buf_names) > 0:
                    get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                        lambda: {
                            "pre_grad_graph_id": V.graph.graph_id,
                            "post_grad_graph_id": V.graph.post_grad_graph_id,
                            "node1_name": node1.get_name(),
                            "node2_name": node2.get_name(),
                            "node1_debug_str": write_text(node1.debug_str()),
                            "node2_debug_str": write_text(node2.debug_str()),
                            "common_buffer_names": list(common_buf_names),
                            "failure_reason": self.decide_fusion_fail_reason(
                                node1, node2, common_buf_names
                            ),
                        }
                    )

                    why("no shared data due to indexing mismatch")
                    return False
            why("no shared data")
            return False  # heuristic not needed for correctness

        if (
            not node1.is_foreach()
            and not node2.is_foreach()
            and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
        ):
            why("exceeds max fusion")
            return False  # heuristic not needed for correctness

        if node1.get_operation_names() & node2.ancestors:
            # node2 depends on node1 outputs
            if not self.can_fuse_vertical(node1, node2):
                return False
            return self.get_backend(device).can_fuse_vertical(node1, node2)
        else:  # nodes don't depend on each other, but may have common reads
            if self.can_fusion_increase_peak_memory(node1, node2):
                why("will increase peak memory")
                return False
            return self.get_backend(device).can_fuse_horizontal(node1, node2)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.
        """
        node1_buf_names = node1.get_buffer_names()
        why = WhyNoFuse(node1, node2)
        remaining_deps_by_name: Dict[str, List[Dep]] = defaultdict(list)

        for dep in node2.unmet_dependencies:
            name = self.mutation_renames.get(dep.name, dep.name)
            if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
                continue
            remaining_deps_by_name[name].append(dep)

        for cd in node1.read_writes.writes:
            if not isinstance(cd, MemoryDep):
                continue
            remaining = remaining_deps_by_name.get(
                self.mutation_renames.get(cd.name, cd.name)
            )
            if remaining:
                for rd in remaining:
                    if self.fusable_read_and_write(rd, cd):
                        remaining.remove(rd)

        remaining_deps = OrderedSet(
            [
                dep.name
                for dep in itertools.chain.from_iterable(
                    remaining_deps_by_name.values()
                )
            ]
        )

        if remaining_deps & node1_buf_names:
            # MemoryDeps didn't match and read different locations of the same buffer.
            # Examples here include:
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            why("memory deps did not match")
            return False

        node1_op_names = node1.get_operation_names()
        for name in remaining_deps:
            op_name = self.name_to_buf[name].defining_op.get_name()
            if node1_op_names & self.name_to_fused_node[op_name].ancestors:
                why("intermediate nodes between node1 & node2")
                return False

        return True

    def fusable_weak_dep(
        self, weak_dep: WeakDep, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if weak_dep.name not in node1.get_buffer_names():
            return False

        # A weak dep can be fused if and only if the fused operation acts inplace
        # on the buffer being mutated. i.e. the same index is being read then mutated
        mutating_writes = [
            write
            for write in node2.read_writes.writes
            if write.name == weak_dep.mutating_buf
        ]
        if len(mutating_writes) != 1:
            return False
        write = mutating_writes[0]
        assert isinstance(write, MemoryDep)

        if free_symbol_is_type(write.index, SymT.TMP):
            return False

        real_name = self.mutation_real_name[weak_dep.mutating_buf]
        relevant_reads = [
            read for read in node1.read_writes.reads if read.name == real_name
        ]
        return all(
            isinstance(read, MemoryDep)
            and not free_symbol_is_type(read.index, SymT.TMP)
            and read.index == write.index
            and read.size == write.size
            for read in relevant_reads
        )

    # StarDep doesn't match MemoryDep, different indices don't match
    # However, broadcasting sometimes strips dimensions, and if that's the case
    # we still can match unmet dep
    # if there's indirect indexing, don't match it
    def fusable_read_and_write(self, read: Dep, write: MemoryDep) -> bool:
        if isinstance(read, MemoryDep):
            read_name = self.mutation_renames.get(read.name, read.name)

            if (
                read_name != write.name
                or free_symbol_is_type(read.index, SymT.TMP)
                or free_symbol_is_type(write.index, SymT.TMP)
            ):
                return False

            if config.loop_ordering_after_fusion and read.num_vars != write.num_vars:
                # Need merge loops if we do loop ordering after fusion since
                # we have not merged the loops yet when creating the scheduler
                # nodes.
                read = read.normalize()
                write = write.normalize()

            return (
                read.index == write.index
                and len(read.size) >= len(write.size)
                and read.size[: len(write.size)] == write.size
            )
        elif isinstance(read, StarDep):
            read_name = self.mutation_renames.get(read.name, read.name)
            write_name = self.mutation_renames.get(write.name, write.name)
            if (
                read.mode == write.mode
                and write.mode is not None
                and read_name == write_name
            ):
                return True
        return False

    def score_fusion(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Tuple[bool, bool, int, int]:
        """
        Assign a score (higher comes first) to the fusion of node1
        and node2.  When different fusions conflict with each other,
        this is the way we decide what order to run them in.

        Our current score is based on:
        - Estimate of the saved memory operations
        - Fusions closer together in original order
        """
        memory_score = self.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return (
            node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )

    def dep_size_hint(self, dep: Dep) -> int:
        res = 0
        if dep not in self.__dep_size_hint_cache:
            try:
                if not dep.has_unbacked_symbols():
                    res = dep.numbytes_hint()
            except KeyError:
                # In at least one test (test/inductor/test_torchbind.py) we
                # create a StarDep that doesn't exist in the graph and calling
                # `has_unbacked_symbols()` throws an error.
                pass
            self.__dep_size_hint_cache[dep] = res
        else:
            res = self.__dep_size_hint_cache[dep]
        return res

    def score_fusion_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        """
        The first term in our fusion score that estimates number of saved
        memory operations.
        """
        node1_dep_len = len(node1.read_writes.reads) + len(node1.read_writes.writes)
        node2_dep_len = len(node1.read_writes.reads) + len(node2.read_writes.writes)

        # optimization: iter over smaller set
        if max(node1_dep_len, node2_dep_len) * 4 > min(node1_dep_len, node2_dep_len):
            if node1_dep_len > node2_dep_len:
                tmp = node1
                node1 = node2
                node2 = tmp

            deps = []
            for dep in node1.read_writes.reads | node1.read_writes.writes:
                if dep in node2.read_writes.reads or dep in node2.read_writes.writes:
                    deps.append(dep)

            return sum(self.dep_size_hint(dep) for dep in deps)

        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        return sum(self.dep_size_hint(dep) for dep in common_memory_deps)

    def get_possible_fusions_with_highest_priority(
        self, possible_fusions: List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
    ) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        # Group the possible fusions based on their priority from the backend.
        # Only return the group of possible fusions with highest priority.
        if len(possible_fusions) == 0:
            return possible_fusions
        possible_fusions_group_by_priority: Dict[
            int, List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
        ] = {}

        for node1, node2 in possible_fusions:
            assert node1.get_device() == node2.get_device()
            device = node1.get_device()
            fusion_pair_priority = int(
                self.get_backend(device).get_fusion_pair_priority(node1, node2)
            )
            if fusion_pair_priority not in possible_fusions_group_by_priority:
                possible_fusions_group_by_priority[fusion_pair_priority] = [
                    (node1, node2),
                ]
            else:
                possible_fusions_group_by_priority[fusion_pair_priority].append(
                    (node1, node2)
                )
        # return the possible fusions with highest priority
        possible_fusions_with_highest_priority = min(
            possible_fusions_group_by_priority.items(), key=operator.itemgetter(0)
        )[1]
        assert len(possible_fusions_with_highest_priority) > 0
        return possible_fusions_with_highest_priority

    def score_fusion_key(
        self, nodes: Tuple[BaseSchedulerNode, BaseSchedulerNode]
    ) -> Tuple[bool, bool, int, int]:
        """
        Shim for list.sort(key=...)
        """
        node1, node2 = nodes
        return self.score_fusion(node1, node2)

    def compute_last_usage(self) -> None:
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

        future_used_buffers: OrderedSet[str] = OrderedSet(V.graph.get_output_names())

        for node in reversed(self.nodes):
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)

    def free_buffers(self) -> None:
        """Free any buffers that are no longer needed"""
        for name in sorted(
            self.buffer_names_to_free
            - V.graph.removed_buffers
            - V.graph.wrapper_code.freed
        ):
            if name in self.name_to_buf:
                buf = self.name_to_buf[name]
                if buf.can_free():
                    V.graph.wrapper_code.codegen_free(buf.node)
            elif name in V.graph.graph_inputs:
                storage = V.graph.graph_inputs[name].data
                assert isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
                V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

    def remove_kernel_local_buffers(self) -> None:
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """

        fused_node_names = OrderedSet(
            self.name_to_buf[buf].defining_op.get_name()
            for buf in V.kernel.store_buffer_names
            if buf in self.name_to_buf
        )
        names_to_remove = []
        for out_buf in V.kernel.store_buffer_names:
            if out_buf not in self.name_to_buf:
                # Aux buffers created during kernel codegen
                names_to_remove.append(out_buf)
                continue
            users = self.name_to_buf[out_buf].users
            assert users is not None
            users = OrderedSet(user.get_name() for user in users if not user.is_weak)
            if users.issubset(fused_node_names):
                names_to_remove.append(out_buf)

        def remove_filter(n: str) -> bool:
            return (
                n not in V.kernel.must_keep_buffers
                and n not in V.kernel.args.input_buffers
                and n not in self.mutation_renames
                and n not in self.mutation_real_name
            )

        names_to_remove = list(filter(remove_filter, names_to_remove))

        for name in names_to_remove:
            if name in V.kernel.args.inplace_buffers:
                buf = V.kernel.args.inplace_buffers[name]
                if isinstance(buf, str) and buf.startswith("REMOVED"):
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                V.kernel.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        V.kernel.args.output_buffers[name] = "REMOVED"
        V.kernel.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        inner_name = V.kernel.args.inplace_buffers[name].inner_name
        V.kernel.args.inplace_buffers[name] = inner_name.replace(
            "in_out_ptr", "REMOVED"
        )
        V.kernel.removed_buffers.add(name)

    def flush(self) -> None:
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None:
        assert isinstance(scheduler_node, ExternKernelSchedulerNode)
        # 'decide_inplace_update' stores the inplace update decisions in
        # the current kernel from where 'allocate' retrieve those decisions.
        # We have to make sure there is a non-NULL kernel handler to store
        # those inplace update decisions.
        counters["inductor"]["extern_calls"] += 1
        with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
            scheduler_node.decide_inplace_update()
            scheduler_node.mark_run()
        node = scheduler_node.node
        assert isinstance(node, ir.ExternKernel), f"{type(node)=}"
        node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    def create_backend(self, device: torch.device) -> BaseScheduling:
        assert (
            not is_gpu(device.type) or device.index is not None
        ), f"{device} should have been normalized in lowering"
        V.graph.add_device_info(device)

        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise RuntimeError(
                    f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
                )
            elif is_gpu(device.type):
                raise RuntimeError(
                    "Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
                )

        return device_scheduling(self)

    def get_backend(self, device: torch.device) -> BaseScheduling:
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def enter_context(self, node: BaseSchedulerNode) -> None:
        def get_order(n: torch.fx.Node) -> int:
            if n not in self.origin_to_index:
                self.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return self.origin_to_index[n]

        # Use a dict to have ordering
        origins = {
            (get_order(e), e): None
            for n in node.get_nodes()
            if n.node is not None
            for e in n.node.get_origins()
        }
        origins = list(origins.keys())
        if origins:
            _, last = max(origins, key=operator.itemgetter(0))
            V.graph.wrapper_code.enter_context(last)

    def codegen(self) -> None:
        with dynamo_timed("Scheduler.codegen"):
            return self._codegen()

    def _codegen(self) -> None:
        if config.check_stack_no_cycles_TESTING_ONLY:
            import torch._dynamo.convert_frame

            stack = traceback.extract_stack()
            seen = OrderedSet[tuple[str, Union[int, None]]]()
            for frame in reversed(stack):
                # This is where maybe_cprofile is
                if (
                    frame.name == "_compile_inner"
                    and frame.filename == torch._dynamo.convert_frame.__file__
                ):
                    break
                key = (frame.filename, frame.lineno)
                assert key not in seen, (
                    f"Duplicate stack frame {frame.filename}:{frame.lineno}; "
                    "did you add a decorator to one of the functions in this stack "
                    "trace?  If so, try using a context manager instead."
                )
                seen.add(key)

        self.current_device = None
        for node in self.nodes:
            if log.isEnabledFor(logging.DEBUG):
                try:
                    log.debug(
                        "Generating code for node %s with estimated runtime %f",
                        node.get_name(),
                        node.get_estimated_runtime(),
                    )
                except Exception as e:
                    log.debug(
                        "Generating code for node %s with estimated runtime 0.0",
                        node.get_name(),
                    )

            self.enter_context(node)

            if not isinstance(node, NopKernelSchedulerNode) and (
                device := node.get_device()
            ):
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if self.current_device and device_need_guard(
                        self.current_device.type
                    ):
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    self.current_device = device
                    if device_need_guard(device.type):
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_device_guard_enter(device.index)

            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                node = typing.cast(ExternKernelSchedulerNode, node)
                self.codegen_extern_call(node)
            elif node.is_foreach():
                node = typing.cast(ForeachKernelSchedulerNode, node)
                backend_ = self.get_backend(device)
                from .codegen.cuda_combined_scheduling import CUDACombinedScheduling
                from .codegen.simd import SIMDScheduling

                if isinstance(backend_, (SIMDScheduling, CUDACombinedScheduling)):
                    backend = backend_
                else:
                    raise AssertionError(f"{type(self)=}")
                backend.codegen_combo_kernel(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_node(node)
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.mark_run()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_buffer_names())
            self.completed_operations.update(node.get_operation_names())

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if device is not None and self.get_backend(device).ready_to_flush():
                    self.flush()

        if self.current_device and device_need_guard(self.current_device.type):
            # exit the outermost CUDA device guard. this is
            # important for nested indentation codegen-ing.
            V.graph.wrapper_code.codegen_device_guard_exit()

        self.flush()

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        device = node_list[0].get_device()
        V.graph.scheduler = self
        self.current_device = device
        backend = self.get_backend(device)
        return backend.benchmark_combo_kernel(node_list)

    def speedup_by_combo_kernel(self, nodes: List[BaseSchedulerNode]) -> bool:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """
        if not config.benchmark_combo_kernel:
            return True

        subkernel_nodes = nodes
        device = subkernel_nodes[0].get_device()

        # don't support benchmark fusion for CPU right now.
        if device.type == "cpu":
            return True

        from triton.compiler.errors import CompilationError

        ms1, path1_list = 0.0, []
        for i, snode in enumerate(subkernel_nodes):
            node_list = snode.get_nodes()
            # We can not accurately benchmark kernel using atomic_add
            # due to how we generate random integer inputs.
            if self._any_atomic_add(node_list):
                fusion_log.debug(
                    "ComboKernel: benchmarking may not accurate due to atomic_add"
                )

            try:
                ms, path = self.benchmark_fused_nodes(node_list)
                if math.isinf(ms):
                    fusion_log.debug(
                        "ComboKernel benchmark: register spilling of %d-th subkernel",
                        i,
                    )
                    return False
            except CompilationError as e:
                # workaround triton issue: https://github.com/openai/triton/issues/2151
                if "Loop-carried variable" in str(e):
                    fusion_log.debug(
                        "ComboKernel benchmark: return True because of loop-carried variable"
                    )
                    return True  # allow fusion
                else:
                    raise
            ms1 += ms
            path1_list.append(path)

        try:
            ms2, ms2_clone, path2_list = self.benchmark_combo_kernel(subkernel_nodes)
        except CompilationError as e:
            # workaround triton issue: https://github.com/openai/triton/issues/2151
            if "Loop-carried variable" in str(e):
                fusion_log.debug(
                    "ComboKernel benchmark: return True because of loop-carried variable"
                )
                return True  # allow fusion
            else:
                raise

        # small kernels are very likely to have speedup but hard to benchmark. So we skip benchmarking.
        small_kernel = ms2 - ms2_clone < 0.3 or ms1 < 0.3
        if fusion_log.isEnabledFor(logging.DEBUG):
            if ms1 > ms2 or small_kernel:
                fusion_log.debug(
                    "can fuse (benchmark): fusing causes %sx speedup",
                    green_text(f"{ms1 / ms2:.3f}"),
                )
            else:
                fusion_log.debug(
                    "cannot fuse (benchmark): fusing causes %sx slowdown",
                    red_text(f"{ms1 / ms2:.3f}"),
                )
        # ms1 returned by benchmark_fused_nodes discounted clone time
        return ms2 - ms2_clone < ms1 or small_kernel

    def get_buffer_layout(self, buf_name: str) -> ir.Layout:
        buf = self.name_to_buf[buf_name]
        assert buf.node is not None
        return buf.node.get_layout()

    def update_zero_dim_cpu_tensor(self) -> None:
        for node in self.nodes:
            if node.get_device() and is_gpu(node.get_device().type):
                for read in node.read_writes.reads:
                    buffer = V.graph.name_to_buffer.get(read.name)
                    if (
                        buffer
                        and buffer.get_device()
                        and buffer.get_device().type == "cpu"
                        and not isinstance(buffer.layout, MultiOutputLayout)
                        and buffer.get_size() == []
                    ):
                        V.graph.zero_dim_cpu_tensor_list.add(read.name)


class BaseScheduling:
    @classmethod
    def get_backend_features(cls, device: torch.device) -> Sequence[BackendFeature]:
        """Return a set of .codegen.common.BackendFeature()"""
        return ()

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        raise NotImplementedError

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        raise NotImplementedError

    def fuse(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        """
        Fuse two nodes
        """
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        else:
            return FusedSchedulerNode.fuse(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[sympy.Expr]]
    ) -> Tuple[Tuple[sympy.Expr, ...], ...]:
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError

    def codegen_sync(self) -> None:
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
        raise NotImplementedError

    def ready_to_flush(self) -> bool:
        """
        Check whether the backend is requesting the scheduler to flush the generated kernel.
        If not supported, please return False.
        """
        return False

    def flush(self) -> None:
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        raise NotImplementedError

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        """
        Return an unsigned integer which represents the priority of this fusion pair.
        The smaller is with higher priority.
        """
        return 0

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> Tuple[float, float, str]:
        """
        Benchmark the list of nodes to combine and return the execution time
        and memory copy time in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError
