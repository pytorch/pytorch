from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import itertools
import logging
import math
import operator
import os
import pprint
import textwrap
import traceback
import typing
from collections import Counter, defaultdict
from typing import Any, Callable, Generic, Optional, TYPE_CHECKING, TypeVar, Union


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._triton import has_triton

from . import comms, config, dependencies, ir, metrics
from .analyze_preserves_zero_mask import can_codegen_without_upcasts
from .codegen.common import BackendFeature, get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import Dep, MemoryDep, StarDep, WeakDep
from .exc import GPUTooOldForTriton, TritonMissing
from .fx_utils import count_flops_fx, countable_fx
from .ir import (
    get_device_type,
    GraphPartitionSignature,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
)
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
    GraphPartitionMap,
    IndentedBuffer,
    is_collective,
    is_cudagraph_unsafe_op,
    is_gpu,
    is_multi_outputs_template,
    is_output_of_multi_outputs_template,
    is_wait,
    sympy_product,
)
from .virtualized import V


log = logging.getLogger(__name__)
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
loop_ordering_log = torch._logging.getArtifactLogger(__name__, "loop_ordering")

PartitionType = list["BaseSchedulerNode"]


@dataclasses.dataclass
class SchedulerBuffer:
    scheduler: Scheduler
    node: ir.Buffer
    defining_op: Optional[BaseSchedulerNode]
    users: list[NodeUser] = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def defining_op_name(self) -> str:
        op = self.defining_op
        assert op is not None
        return op.get_name()

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

        if (
            self.node.get_inputs_that_alias_output()
            or self.node.get_mutation_names()
            or isinstance(self.node.get_output_spec(), ir.CommBufferLayout)
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # hacky check for if V.kernel is a real kernel or NullHandler
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            input_buffer: Union[ir.DonatedBuffer, ir.Buffer]
            input_buffer_name = V.kernel.inplace_update_buffers[self.get_name()]
            if input_buffer_name in self.scheduler.name_to_donated_buffer:
                input_buffer = self.scheduler.name_to_donated_buffer[
                    input_buffer_name
                ].node
            else:
                input_buffer = self.scheduler.name_to_buf[input_buffer_name].node
            V.graph.wrapper_code.codegen_inplace_reuse(
                input_buffer,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self) -> bool:
        # There's no real allocated buffer, no need to free it
        assert self.node is not None
        if isinstance(self.node.layout, ir.NoneLayout) or is_multi_outputs_template(
            self.node
        ):
            return False
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def set_users(self, users: list[NodeUser]) -> None:
        # deduplicate
        result: dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self) -> Sequence[str]:
        assert self.node is not None
        return self.node.get_inputs_that_alias_output()

    def get_mutations(self) -> Sequence[str]:
        assert self.node is not None
        return self.node.get_mutation_names()

    def get_device(self) -> Optional[torch.device]:
        return self.node.get_output_spec().get_device()


@dataclasses.dataclass
class SchedulerDonatedBuffer(SchedulerBuffer):
    defining_op: Optional[BaseSchedulerNode] = None


class BaseSchedulerNode:
    group: tuple[torch.device, tuple[tuple[sympy.Expr, ...], ...]]
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
        self.debug_device_str: Callable[[BaseSchedulerNode], list[str]] = (
            lambda *args, **kwargs: []
        )

    def _init_from_node(self, node: ir.Operation) -> None:
        self.node: Optional[ir.Operation] = node
        self.ancestors: OrderedSet[str] = OrderedSet()
        self.last_usage = OrderedSet[
            str
        ]()  # buffers that won't be used after this kernel
        self.written = False
        self.outputs: list[SchedulerBuffer] = [
            SchedulerBuffer(
                scheduler=self.scheduler,
                node=output,
                defining_op=self,
            )
            for output in node.get_outputs()
        ]
        self.outputs_by_name: dict[str, SchedulerBuffer] = {
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
{name}: {type(self).__name__}({type(getattr(self, "node", None)).__name__})
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

    def _debug_str_for_device(self) -> list[str]:
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

    def update_mutated_names(self, renames: dict[str, str]) -> None:
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
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
    ) -> None:
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
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
                deps.extend(
                    alias
                    for alias in V.graph.name_to_buffer[
                        dep
                    ].get_inputs_that_alias_output()
                    if alias not in used_names
                )
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
            op_name = self.scheduler.name_to_buf[dep.name].defining_op_name()
            return op_name in V.graph.removed_operations

        to_remove = OrderedSet(
            dep for dep in self.read_writes.reads if should_prune(dep)
        )
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

    def get_name(self) -> str:
        assert self.node is not None
        return self.node.get_operation_name()

    def get_first_name(self) -> str:
        return self.get_name()

    @cache_on_self
    def get_operation_names(self) -> OrderedSet[str]:
        return OrderedSet(node.get_name() for node in self.get_nodes())

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet(out.get_name() for out in self.outputs)

    @cache_on_self
    def can_codegen_in_low_precision(self) -> bool:
        return all(
            isinstance(n, SchedulerNode)
            and can_codegen_without_upcasts(n, disallow_fp32_ops=True)
            for n in self.get_nodes()
        )

    @cache_on_self
    def can_codegen_without_upcasts(self) -> bool:
        return all(
            isinstance(n, SchedulerNode) and can_codegen_without_upcasts(n)
            for n in self.get_nodes()
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return [self]

    def get_outputs(self) -> Sequence[SchedulerBuffer]:
        return self.outputs

    def get_output(self, buf_name: str) -> SchedulerBuffer:
        return self.outputs_by_name[buf_name]

    def get_device(self) -> Optional[torch.device]:
        assert self.node is not None
        return self.node.get_device()

    def is_cpu(self) -> bool:
        device = self.get_device()
        return device is not None and device.type == "cpu"

    def is_gpu(self) -> bool:
        device = self.get_device()
        return device is not None and is_gpu(device.type)

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
        from .codegen.wrapper import can_match_buffer_size

        if not (
            isinstance(self, SchedulerNode)
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

        # NOTE remove V.graph.removed_operations once deps issue is fixed
        inconsequential_nodes = (
            self.ancestors
            | V.graph.removed_operations
            | self.scheduler.completed_operations
        )

        def single_index_in_fused_node(buf_to_be_inplaced: SchedulerBuffer) -> bool:
            # Inside of NodeUser, we track that the read and write are equivalent
            # before deciding if the use can be inplace.
            # But if that use is fused into a larger kernel, we need to check equivalence
            # of other accesses in fused scheduler node as well.
            fused_node = buf_to_be_inplaced.scheduler.get_fused_node(self)
            buf_name = buf_to_be_inplaced.get_name()
            # Dedup read/writes with equivalent indices
            # TODO - would be nice if we could just cache accesses on ReadWrites,
            # and enforce variant that this class & members are functional..
            deps: OrderedSet[Dep] = OrderedSet()
            for user in buf_to_be_inplaced.users:
                user_node = user.node
                if not isinstance(user_node, BaseSchedulerNode):
                    continue

                if (
                    user_node.get_first_name()
                    not in buf_to_be_inplaced.scheduler.name_to_fused_node
                    or buf_to_be_inplaced.scheduler.get_fused_node(user_node)
                    is not fused_node
                ):
                    continue

                deps |= (
                    o
                    for o in user_node.read_writes.reads_and_writes()
                    if o.name == buf_name
                )
                if len(deps) > 1:
                    return False

            return True

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

            for read in self.read_writes.reads:
                input_buf: Optional[Union[SchedulerBuffer, SchedulerDonatedBuffer]]
                if read.name in self.scheduler.name_to_donated_buffer:
                    input_buf = self.scheduler.name_to_donated_buffer[read.name]
                else:
                    input_buf = self.scheduler.name_to_buf.get(read.name)

                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, self)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    assert input_buf.users is not None
                    remaining_uses = [
                        x
                        for x in input_buf.users
                        if x.node.get_name() not in inconsequential_nodes
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and input_buf.node is not None
                        and not isinstance(
                            input_buf.node.get_output_spec(),
                            (
                                ir.NoneLayout,
                                ir.MultiOutputLayout,
                                ir.MutationLayoutSHOULDREMOVE,
                            ),
                        )
                        and not (
                            input_buf.defining_op
                            and isinstance(
                                input_buf.defining_op.node,
                                (ir.FallbackKernel, ir.MultiOutput),
                            )
                            and len(input_buf.node.get_inputs_that_alias_output()) > 0
                        )
                        and can_match_buffer_size(input_buf.node, buf.node)
                        and single_index_in_fused_node(input_buf)
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

                        V.kernel.inplace_update_buffers[buf.get_name()] = (
                            input_buf.get_name()
                        )
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
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=True
        )

    @cache_on_self
    def get_read_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=False
        )

    @cache_on_self
    def get_write_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=False, include_writes=True
        )

    def get_read_write_buffers_sizes_impl(
        self, include_reads: bool, include_writes: bool
    ) -> int:
        return sum(
            self.get_read_write_buffer_accesses(
                include_reads=include_reads, include_writes=include_writes
            ).values(),
            start=0,
        )

    def get_read_write_buffer_accesses(
        self, include_reads: bool, include_writes: bool
    ) -> dict[str, int]:
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

        Returns memory accesses per buffer.
        """
        if isinstance(self, NopKernelSchedulerNode):
            return {}
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(
            self.node, MultiOutput
        ):
            # todo: Calculate this - it's kinda annoying.
            return {}
        if (
            isinstance(self, ExternKernelSchedulerNode)
            and isinstance(self.node, ir.FallbackKernel)
            and self.node.op_overload
            is torch._prims.rng_prims.graphsafe_run_with_rng_state
        ):
            return {}

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

        if include_reads:
            for dep in self.read_writes.reads:
                buf_accesses[dep.name].append(dep)

        if include_writes:
            for dep in self.read_writes.writes:
                buf_accesses[dep.name].append(dep)

        reads = (
            OrderedSet(dep.name for dep in self.read_writes.reads)
            if include_reads
            else OrderedSet()
        )
        writes = (
            OrderedSet(dep.name for dep in self.read_writes.writes)
            if include_writes
            else OrderedSet()
        )

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

        buf_byte_accesses: dict[str, int] = {}

        for buf_name in reads | writes:
            buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
            buf: Union[ir.Buffer, ir.TensorBox, ir.TorchBindObject]
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_bytes(
                buf: Optional[Union[ir.Buffer, ir.TensorBox, ir.TorchBindObject]],
            ) -> int:
                if not buf:
                    return 0

                if isinstance(buf, ir.TorchBindObject):
                    return buf.get_buf_bytes()
                elif isinstance(buf.layout, MultiOutputLayout):
                    # Kind of a lazy way to get the MultiOutput nodes corresponding to
                    # a MultiOutputLayout
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

            buf_bytes = get_buf_bytes(buf)
            if buf_name not in buf_byte_accesses:
                buf_byte_accesses[buf_name] = buf_bytes
            else:
                buf_byte_accesses[buf_name] += buf_bytes

        return buf_byte_accesses

    @cache_on_self
    def estimate_flops(self) -> int | None:
        if self.node is None:
            return None
        fx_node = self.node.get_origin_node()
        if fx_node is None:
            return None
        if not countable_fx(fx_node):
            return None

        flops = count_flops_fx(fx_node)

        resolved_flops = V.graph.sizevars.size_hints((flops,), fallback=0)[0]
        counters["inductor"]["flop_count"] += resolved_flops
        return resolved_flops

    @cache_on_self
    def get_estimated_runtime(self) -> float:
        """
        Returns estimated op runtime in nanoseconds (ns)
        """
        buf = self.get_nodes()[0].get_outputs()[0]
        layout = buf.node.get_output_spec()
        if not is_gpu(get_device_type(layout)):
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
            except TypeError as e:
                # this happens when the collective is not of type ir._CollectiveKernel
                log.info(e)
                return 0

        elif is_wait(self.node):
            # ir.Wait is only used for collective ops.
            # The time needed for the collective op is already estimated and considered
            # when we are processing the collective op IR node, so ir.Wait takes 0 time
            # since it doesn't take extra time to get the result after the collective is completed.
            return 0

        dtype = buf.node.maybe_get_dtype()
        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
            # If cudaGetDeviceProperties returns 0 for gpu_memory_bandwidth or gpu_flops
            # there is a chance to continue execution successfully. Otherwise, it would fail with
            # ZeroDivisionError below.
            if gpu_memory_bandwidth <= 0:
                raise AssertionError(
                    f"gpu_memory_bandwidth cannot be <= 0, but got {gpu_memory_bandwidth}"
                )
            if gpu_flops <= 0:
                raise AssertionError(f"gpu_flops cannot be <= 0, but got {gpu_flops}")
        except Exception:
            return 0

        flops_est = self.estimate_flops()

        if flops_est == 0 or flops_est is None:
            # no flops estimate, so fall back to memory estimate
            return self.get_read_write_buffers_sizes() / gpu_memory_bandwidth

        # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
        factor = 1.0
        counted_bytes = self.get_read_write_buffers_sizes()
        counted_bytes = 0 if counted_bytes is None else counted_bytes
        compute_time = (factor * flops_est / gpu_flops) * 1e9
        transfer_time = counted_bytes / gpu_memory_bandwidth

        # Return estimated runtime in nanoseconds
        return max(compute_time, transfer_time)

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return None

    def get_template_node_or_throw(self) -> ir.TemplateBuffer:
        template = self.get_template_node()
        assert template is not None
        return template

    @staticmethod
    def get_prologue_template_epilogue(
        nodes: list[BaseSchedulerNode],
    ) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
        """
        For the list of nodes, get the prologue, template, and epilogue
        """
        template_index = next(i for i, n in enumerate(nodes) if n.is_template())

        prologue = nodes[:template_index]
        template_node = nodes[template_index]
        epilogue = nodes[template_index + 1 :]
        return prologue, template_node, epilogue


class WhyNoFuse:
    # TODO when we drop support for Python < 3.10, we can use
    # @dataclass(slots=True) instead of manually specifying __slots__.
    __slots__ = ["name1", "name2", "reason", "args"]
    reason: str
    args: tuple[Any, ...]

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        self.name1 = node1.get_name()
        self.name2 = node2.get_name()

    def __call__(self, reason: str, *args: Any) -> None:
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self) -> str:
        return f"cannot fuse {self.name1} with {self.name2}: " + (
            self.reason % self.args
        )


def pformat(obj: Any) -> str:
    if isinstance(obj, (OrderedSet, set)):  # noqa: set_linter
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
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
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
            op_name = name_to_buf[dep.name].defining_op_name()
            name_to_dep_count[name_to_fused_node[op_name].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op_name()
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
    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody

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
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))
        self._sizes, self._body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Don't normalize since normalization will merge loops which
        # makes it hard to decide new loop orders.
        should_normalize = not config.loop_ordering_after_fusion or not is_gpu(
            device.type
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
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._compute_attrs(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )

    def refresh_dependencies(
        self, normalize: bool, need_clear_tiling_cache: bool
    ) -> None:
        # Fake dependencies are added manually. They can not be analyzed from
        # extract_read_writes. Find them out and apply manually.
        fake_deps: OrderedSet[Dep] = OrderedSet(
            dep for dep in self.read_writes.reads if isinstance(dep, (WeakDep, StarDep))
        )

        # don't normalize since the loop order may need to be further changed
        # later
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=normalize
            ).with_read(fake_deps)
        )

        self.pointwise_read_writes.clear_cache(self)

        if need_clear_tiling_cache:
            from .codegen.simd import SIMDScheduling

            # TODO(shunting) if this cause compilation time increase when
            # enabling LOAF by default, try just clearing the specific cache
            # entry by using a customized cache implementation rather than
            # lru_cache.
            SIMDScheduling.candidate_tilings.cache_clear()

    def apply_new_loop_order(self, new_order: Sequence[int]) -> None:
        self._body = self._body.reorder_iter_loops(
            new_order,
        )
        self._sizes = self._body.sizes

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def merge_loops(self) -> None:
        self._body = self._body.merge_loops()
        self._sizes = self._body.sizes

        # merge_loops is called after loop reordering.
        # We still need retain fake dependencies since codegen the
        # estimated amount of memory access rely on them.
        #
        # Merge loops does not affect the tiling decision. So we
        # don't need clear the tiling cache.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)

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
                if not isinstance(buf, ir.TorchBindObject):
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
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)), (
            f"{type(self.node)=}"
        )
        return bool(self.node.get_reduction_type())

    def is_split_scan(self) -> bool:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)), (
            f"{type(self.node)=}"
        )
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
    ) -> dict[sympy.Expr, sympy.Expr]:
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
            with (
                V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)),
                V.kernel.set_current_node(self),
            ):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_or_reduction_read_writes(
        self, pointwise: bool = True
    ) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in either the pointwise or the reduction axes.
        """
        keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
        return dependencies.extract_read_writes(
            self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
        )

    @cache_on_self
    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the non-reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=True)

    @cache_on_self
    def reduction_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=False)

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


def refresh_group_node_dependencies(
    group_snode: Union[FusedSchedulerNode, GroupedSchedulerNode],
) -> None:
    snodes = group_snode.snodes
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
    group_snode: Union[FusedSchedulerNode, GroupedSchedulerNode],
    scheduler: Scheduler,
    snodes: list[BaseSchedulerNode],
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

    snodes: list[BaseSchedulerNode]

    @classmethod
    def fuse(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        assert node1.scheduler is node2.scheduler
        assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
        if node1.is_template() and isinstance(node2, ExternKernelSchedulerNode):
            # Fuse multi outputs template and its outputs
            #   * Node1 has memorydep of MultiOutput in reads
            #   * Node2 has StarDep of MultiOutput in writes
            # Rewrite the Node2' StarDep to MemoryDep, because calculate score_fusion_memory
            # of the template node and its epilogue requires the same type of dependencies
            assert isinstance(node2.node, MultiOutput)
            assert len(node2.read_writes.writes) == 1
            assert isinstance(next(iter(node2.read_writes.writes)), StarDep)
            name = next(iter(node2.read_writes.writes)).name
            template_nodes = [node for node in node1.get_nodes() if node.is_template()]
            assert len(template_nodes) == 1
            template_node = template_nodes[0]
            assert len(template_node.read_writes.writes) == 1
            write = next(iter(template_node.read_writes.writes))
            assert isinstance(write, MemoryDep)
            node2.read_writes.writes = OrderedSet(
                [
                    MemoryDep(
                        name, write.index, write.var_names, write.size, write.mode
                    ),
                ]
            )
        else:
            assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> None:
        if self.is_template():
            # We can not really reorder loops for a triton template
            return
        self_sizes = None
        for snode in self.snodes:
            assert isinstance(snode, SchedulerNode)
            if self_sizes is not None and tuple(self_sizes) != tuple(snode._sizes[0]):
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
            snode.apply_new_loop_order(new_order)

        refresh_group_node_dependencies(self)

    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        self.users: list[NodeUser] = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list[SchedulerBuffer]:
        result: list[SchedulerBuffer] = []
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
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
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
    def update_mutated_names(self, renames: dict[str, str]) -> None:
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

            node_name = self.scheduler.name_to_buf[rd.name].defining_op_name()
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
        fused_nodes: list[BaseSchedulerNode]
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
        snodes: list[BaseSchedulerNode],
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
            self.users: list[NodeUser] = []

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

            self.outputs_by_name: dict[str, SchedulerBuffer] = {
                k: v for snode in self.snodes for k, v in snode.outputs_by_name.items()
            }

        self.use_custom_partition_algo = use_custom_partition_algo
        device = snodes[0].get_device()
        assert device
        self.group = (device, ((sympy.Expr("combo_kernel"),),))
        self.origins = OrderedSet[torch.fx.Node]()
        self.enable_autotune = enable_autotune

    @classmethod
    def combinable_nodes(
        cls, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
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
                "ComboKernels: %d template nodes are filtered: %s",
                len(template_nodes),
                template_nodes,
            )
        filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]
        return filtered_nodes

    @staticmethod
    def _default_group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
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
        [Scheduler], list[list[BaseSchedulerNode]]
    ] = _default_group_nodes_for_combo_kernels

    @staticmethod
    def set_group_algorithm_for_combo_kernels(
        custom_group_algorithm: Callable[[Scheduler], list[list[BaseSchedulerNode]]],
    ) -> None:
        ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels = (
            custom_group_algorithm
        )

    @staticmethod
    def group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
        return ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels(scheduler)

    def mark_run(self) -> None:
        raise NotImplementedError

    def codegen(self) -> None:
        raise NotImplementedError

    def is_foreach(self) -> bool:
        return True

    def get_subkernel_nodes(self) -> list[BaseSchedulerNode]:
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
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
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

    snodes: list[BaseSchedulerNode]

    @classmethod
    def create(cls, snodes: list[BaseSchedulerNode]) -> GroupedSchedulerNode:
        scheduler = snodes[0].scheduler
        assert all(node.scheduler is scheduler for node in snodes)
        grouped_snode = cls(scheduler, snodes)
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped_snode
        scheduler.name_to_fused_node[grouped_snode.get_name()] = grouped_snode
        return grouped_snode

    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)

    def unpack(self) -> list[BaseSchedulerNode]:
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

    def get_outputs(self) -> list[SchedulerBuffer]:
        result: list[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return self.snodes

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        # GroupedSchedulerNode cannot be fused with another node
        return False


def pick_loop_order(
    stride_lengths: list[list[int]],
    sizes: Sequence[sympy.Expr],
    priority_idx: tuple[int, ...] = (),
) -> list[int]:
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
    """
    A Scheduler is a graph of BaseSchedulerNodes. It is responsible for
    optimizations such as fusion, reorder, and graph partition.
    """

    __dep_size_hint_cache: dict[Dep, int]

    def __init__(self, nodes: list[ir.Operation]) -> None:
        with dynamo_timed("Scheduler.__init__"):
            self._init(nodes)

    def _init(self, nodes: list[ir.Operation]) -> None:
        super().__init__()
        self.__dep_size_hint_cache = {}
        V.graph.scheduler = self
        self.backends: dict[torch.device, BaseScheduling] = {}
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self._graph_partition_counter = itertools.count()

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

        self.name_to_donated_buffer: dict[str, SchedulerDonatedBuffer] = (
            self.get_donated_buffers()
        )
        self.name_to_node: dict[str, BaseSchedulerNode] = {
            n.get_name(): n for n in self.nodes
        }
        self.name_to_buf: dict[str, SchedulerBuffer] = {
            buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
        }
        self.name_to_fused_node: dict[str, BaseSchedulerNode] = self.name_to_node.copy()

        # mutation_real_name: Maps back to the original name for codegen
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_real_name = {"buf0" : "buf1"}
        # all subsequent uses of buf0 become buf1's usage in dependency graph
        self.mutation_real_name: dict[str, str] = {}

        # We handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_renames = {"buf1" : "buf0"}
        # in codegen we only use buf0, never buf1
        self.mutation_renames: dict[str, str] = {}

        # Must run first to correctly set dependencies, before all other passes that rely on
        # reading from .read_writes.reads or .unmet_dependencies
        self.nodes = comms.decide_global_ordering_of_comms(
            self.nodes,
            self.name_to_buf,
            self.name_to_fused_node,
        )

        self.compute_dependencies()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.dead_node_elimination()
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.compute_ancestors()

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        from torch._inductor.debug import log_ir_post_fusion, log_ir_pre_fusion

        log_ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.create_foreach_nodes()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.logged_slow_fusion = OrderedSet[tuple[str, str]]()
        if config._pre_fusion_custom_pass is not None:
            self.nodes = config._pre_fusion_custom_pass(self.nodes)
        self.nodes = self.fuse_nodes(self.nodes)
        if config._post_fusion_custom_pass is not None:
            self.nodes = config._post_fusion_custom_pass(self.nodes)
        self.merge_loops()
        self.finalize_multi_template_buffers()
        if config.combo_kernels:
            self.create_combo_kernel_nodes(num_ck_nodes=None)

        # Peak memory pass and overlap pass must run last, otherwise
        # other reordering passes could undo their effects.
        if config.reorder_for_peak_memory:
            from .memory import reorder_for_peak_memory

            self.nodes = reorder_for_peak_memory(
                self.nodes,
                self.name_to_buf,
                self.name_to_fused_node,
                OrderedSet(V.graph.graph_inputs.keys()),
                OrderedSet(V.graph.get_output_names()),
            )
        if config.reorder_for_compute_comm_overlap:
            self.nodes = comms.reorder_compute_and_comm_for_overlap(self.nodes)
        self.process_grouped_nodes()

        if torch._inductor.config.graph_partition:
            self.nodes = self.maybe_reorder_for_minimizing_partition(self.nodes)
            self.nodes = self.reorder_for_partition_with_simple_dependency(self.nodes)

        self.compute_last_usage()
        log_ir_post_fusion(self.nodes)
        V.debug.graph_diagram(self.nodes)
        self.debug_draw_graph()

        # used during codegen:
        self.buffer_names_to_free: OrderedSet[str] = OrderedSet()

        # fx graph node to the position it appears in the graph
        # for debug attribution
        self.origin_to_index: dict[torch.fx.Node, int] = {}

        get_metric_table("graph_stats").add_row(
            lambda: {
                "graph_id": self.post_grad_graph_id,
                "num_nodes_before_fusion": self.num_orig_nodes,
                "num_nodes_after_fusion": len(self.nodes),
            }
        )

    def get_donated_buffers(self) -> dict[str, SchedulerDonatedBuffer]:
        name_to_donated_buf = {}
        for name in V.graph.graph_inputs_original:
            if isinstance(V.graph.graph_inputs_original[name], ir.DonatedBuffer):
                name_to_donated_buf[name] = SchedulerDonatedBuffer(
                    self,
                    V.graph.graph_inputs_original[name],
                    defining_op=None,
                )
        return name_to_donated_buf

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
        assert node.get_origins() is not None, (
            "All nodes passed to scheduling must have an origin"
        )
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
                items: Optional[list[T]] = None,
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

        name_to_users: defaultdict[str, DedupList[NodeUser]] = collections.defaultdict(
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

        unbacked_symbol_to_origin_node: dict[sympy.Symbol, Optional[str]] = {}

        # NB: None means that the dependency is on an input.  Don't actually
        # generate a dependency because if we do, Inductor will start trying
        # to free the unbacked int but that's pointless
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                for fs in val.free_symbols:
                    unbacked_symbol_to_origin_node[fs] = None
            elif isinstance(val, ir.TensorBox):
                # We also need to add symbols from input size as well because
                # AOTI doesn't lift the unbacked symints to inputs
                sym_size = [s for s in val.get_size() if isinstance(s, sympy.Expr)]
                for s in sym_size:
                    for fs in s.free_symbols:
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
                node.node.get_free_symbol_uses(unbacked_only=True), key=lambda x: x.name
            )
            # if a kernel takes unbacked symints, register dependencies
            for s in unbacked_symbol_uses:
                assert s in unbacked_symbol_to_origin_node, (
                    f"{s} not in {unbacked_symbol_to_origin_node}"
                )
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
                    self.mutation_real_name[buf.get_name()] = (
                        self.mutation_real_name.get(alt_name, alt_name)
                    )

        # make sure outputs aren't dead-code-eliminated
        for buf_name in V.graph.get_output_names():
            log.debug("scheduling output %s", buf_name)
            add_user(buf_name, OutputNode(StarDep(buf_name)))

        # make sure unbacked symints aren't dead-code-eliminated
        for out in V.graph.graph_outputs:
            for s in out.get_free_symbol_uses(unbacked_only=True):
                assert s in unbacked_symbol_to_origin_node, (
                    f"{s} not in {unbacked_symbol_to_origin_node.keys()}"
                )
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

        for name in self.name_to_donated_buffer:
            self.name_to_donated_buffer[name].set_users(name_to_users[name].items)

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
                for read in node.read_writes.reads:
                    if read.name in self.name_to_buf:
                        users = self.name_to_buf[read.name].users
                        self.name_to_buf[read.name].users = [
                            u for u in users if u.node.get_name() != node.get_name()
                        ]
        self.nodes = list(reversed(updated_nodes))

        # Prune any WeakDeps no longer needed
        for node in self.nodes:
            node.prune_weak_deps()

    def topological_sort_schedule(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        """
        Ensure nodes is in topologically sorted order
        """
        seen = OrderedSet[BaseSchedulerNode]()
        name_to_node: dict[str, BaseSchedulerNode] = dict()
        result: list[BaseSchedulerNode] = []

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

    def _get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        unmet_deps: OrderedSet[str] = OrderedSet()
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
        unmet_dep_ops = (self.name_to_buf[dep].defining_op_name() for dep in unmet_deps)
        return list(OrderedSet(self.name_to_fused_node[n] for n in unmet_dep_ops))

    def _topological_sort_nodes(self) -> list[list[BaseSchedulerNode]]:
        """
        Sort nodes by their topological order, return a list of node lists.
        """
        order = []
        nodes = dict.fromkeys(self.nodes, 0)
        children: dict[Any, Any] = {}
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
        name_to_ancestors: dict[str, OrderedSet[str]] = {}
        for node in self.nodes:
            ancestors: OrderedSet[str] = OrderedSet()
            for dep in node.unmet_dependencies:
                dep_node_name = self.name_to_buf[dep.name].defining_op_name()
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
                not node.is_gpu() and config.cpu_backend != "halide"
            ):
                continue
            for snode in node.get_nodes():
                # merge loops for the scheduler node
                if not isinstance(snode, SchedulerNode) or snode.is_template():
                    continue

                snode.merge_loops()

                # Note that for CPU backend, merging loops will change
                # snode.group. It's fine for Triton backend.
                # But if we simplify update snode.group like this:
                #   group_fn = self.get_backend(snode.node.get_device()).group_fn
                #   snode.group = (snode.node.get_device(), group_fn(snode._sizes))
                # There is still an issue due to different snode in a
                # FusedSchedulerNode having different merged loops.
                # Skip CPU backend for now.

    def fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.
        """
        with dynamo_timed(
            "Scheduler.fused_nodes", log_pt2_compile_event=True, log_waitcounter=True
        ):
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
                    fusion_log.debug(
                        "===== fusion complete (%d iterations) =====", i + 1
                    )
                    break
            return nodes

    def process_grouped_nodes(self) -> None:
        """
        Unpack GroupedSchedulerNode into regular nodes.
        """
        new_nodes: list[BaseSchedulerNode] = []
        for node in self.nodes:
            new_nodes.extend(
                node.unpack() if isinstance(node, GroupedSchedulerNode) else [node]
            )
        self.nodes = new_nodes

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        assert len(nodes) > 0
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed(
            "benchmark_fused_nodes",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="compile_time_autotune_time_us",
        ):
            return backend.benchmark_fused_nodes(nodes)

    def generate_kernel_code_from_nodes(
        self, nodes: Sequence[BaseSchedulerNode], benchmark_kernel: bool
    ) -> str:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        assert len(nodes) > 0
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed("benchmark_fused_nodes"):
            return backend.generate_kernel_code_from_nodes(nodes, benchmark_kernel)

    def benchmark_codegened_module(
        self, module: ModuleType, device: torch.device
    ) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed("benchmark_fused_nodes"):
            return backend.benchmark_codegened_module(module)

    def finalize_multi_template_buffers(self) -> None:
        """
        Finalize a backing choice for MultiTemplateBuffers which did not already have a
        choice finalized through fusion. In the case of an extern choice, this will result
        in replacing the SchedulerNode.

        If a MultiTemplateBuffer did not have any fusion opportunities, finalizing a choice
        will force completion of compilation and benchmarking.
        """

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
                    )

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

                # We need to reflect the mutation renames that were recorded in the original node
                mutation_renames = {}
                for dep in itertools.chain(
                    node.read_writes.reads, node.unmet_dependencies
                ):
                    if real_name := self.mutation_real_name.get(dep.name, None):
                        mutation_renames[real_name] = dep.name

                def rename_deps(deps: OrderedSet[Dep]) -> OrderedSet[Dep]:
                    return OrderedSet(dep.rename(mutation_renames) for dep in deps)

                new_scheduler_node.unmet_dependencies = rename_deps(
                    new_scheduler_node.unmet_dependencies
                )
                new_scheduler_node.read_writes.reads = rename_deps(
                    new_scheduler_node.read_writes.reads
                )

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
    ) -> Union[bool, Callable[[], bool]]:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """

        is_multi_template = any(
            n.is_template()
            and isinstance(n.get_template_node(), ir.MultiTemplateBuffer)
            for n in (node1, node2)
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
        assert device

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

        device = node_list_fused[0].get_device()
        assert device is not None

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

        async_compile = torch._inductor.async_compile.AsyncCompile()

        def compile_kernel(
            nodes: Sequence[BaseSchedulerNode],
        ) -> tuple[Optional[LambdaFuture], ModuleType]:
            src_code = self.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel=True
            )
            mod = PyCodeCache.load(src_code)
            if not async_compile.use_process_pool():
                fut = None
            else:
                fut = async_compile.triton(kernel_name="triton_", source_code=src_code)
                assert isinstance(fut, LambdaFuture)

            return (fut, mod)

        if is_multi_template and any(
            n.get_template_node() is not None for n in (node1, node2)
        ):
            epilogue_fusion = node1.get_template_node() is not None
            multi_node = (
                node1.get_template_node()
                if epilogue_fusion
                else node2.get_template_node()
            )
            assert isinstance(multi_node, ir.MultiTemplateBuffer)

            # Eagerly compile and benchmark non-template nodes
            choice_timings = multi_node.choice_timings
            _, ms1 = multi_node.get_min_choice()
            ms2, path2 = (
                self.benchmark_fused_nodes(node_list_2)
                if epilogue_fusion
                else self.benchmark_fused_nodes(node_list_1)
            )

            # Start compiling choices in parallel
            future_choices: list[tuple[Any, Optional[LambdaFuture], ModuleType]] = []
            triton_choices = 0
            for choice, unfused_time in sorted(
                choice_timings.items(), key=operator.itemgetter(1)
            ):
                if not isinstance(choice, torch._inductor.ir.TritonTemplateCallerBase):
                    continue

                # For prologue fusion we check if the underlying template of the choice
                # supports all allowed prologue inputs. If not, we skip this choice in
                # the fusion benchmark.
                # TODO: Remove this check after all Triton templates support prologue fusion.
                # Currently, persistent+TMA Triton template does not due to the TMA-based loads.
                if (
                    not epilogue_fusion
                    and hasattr(choice, "allowed_prologue_inps")
                    and choice.allowed_prologue_inps != multi_node.allowed_prologue_inps
                ):
                    continue

                if unfused_time >= ms1 + ms2:
                    break

                triton_choices += 1
                if triton_choices > config.max_epilogue_benchmarked_choices:
                    break

                with multi_node.swap_as_triton_caller(choice):
                    future_choices.append((choice, *compile_kernel(node_list_fused)))

            if len(future_choices) == 0:
                return False

            def benchmark_when_ready() -> bool:
                min_ms_fused = float("inf")
                ms_fused_choice = None

                new_timings = {}
                # Benchmark each choice after compilation completes
                for choice, future, mod_fused in future_choices:
                    try:
                        if future is not None:
                            future.result()

                    # Ideally we would more narrowly catch Exceptions here but
                    # triton  will unpredictably error with valid prologue fusions
                    except Exception as e:
                        if fusion_log.isEnabledFor(logging.DEBUG):
                            fusion_log.debug(
                                "Exception in compiling %s: %s",
                                "prologue" if not epilogue_fusion else "epilogue",
                                str(e),
                            )
                        continue
                    with multi_node.swap_as_triton_caller(choice):
                        ms_fused, path = self.benchmark_codegened_module(
                            mod_fused, device
                        )
                        new_timings[choice] = ms_fused
                        if ms_fused < min_ms_fused:
                            min_ms_fused = ms_fused
                            ms_fused_choice = choice

                log_fusion(min_ms_fused, ms1, ms2)

                if min_ms_fused < (ms1 + ms2) and ms_fused_choice is not None:
                    multi_node.finalize_as_triton_caller(ms_fused_choice)
                    multi_node._choice_timings = new_timings
                    return True
                else:
                    return False

            return benchmark_when_ready

        else:
            # Start parallel compilation for all three kernels
            future_and_mod_l1 = compile_kernel(node_list_1)
            future_and_mod_l2 = compile_kernel(node_list_2)
            future_and_mod_l1_fused = compile_kernel(node_list_fused)

            def benchmark_when_ready() -> bool:
                from torch._inductor.runtime.triton_heuristics import (
                    NoTritonConfigsError,
                )

                try:
                    # Wait for all compilations to complete
                    for fut in (
                        future_and_mod_l1[0],
                        future_and_mod_l2[0],
                        future_and_mod_l1_fused[0],
                    ):
                        if fut is not None:
                            fut.result()

                    ms1, path1 = self.benchmark_codegened_module(
                        future_and_mod_l1[1], device
                    )
                    if math.isinf(ms1):
                        why("register spilling of the first kernel")
                        return False

                    ms2, path2 = self.benchmark_codegened_module(
                        future_and_mod_l2[1], device
                    )
                    if math.isinf(ms2):
                        why("register spilling of the second kernel")
                        return False

                    ms_fused, path_fused = self.benchmark_codegened_module(
                        future_and_mod_l1_fused[1], device
                    )
                    if math.isinf(ms_fused):
                        why("register spilling of the fused kernel")
                        return False

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

                except NoTritonConfigsError:
                    return False

                except CompilationError as e:
                    if "Loop-carried variable" in str(e):
                        return True
                    raise

            return benchmark_when_ready

    def get_fused_node(self, node: BaseSchedulerNode) -> BaseSchedulerNode:
        "Look up the node in Scheduler name_to_fused_node"
        return self.name_to_fused_node[node.get_first_name()]

    def fuse_nodes_once(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
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
                fusion_log.debug("  %s", node.debug_str_short())

        # These are potential fusions which we are async compiling,
        # and which we will benchmark profitability of.
        pending_fusions: dict[
            BaseSchedulerNode,
            tuple[Callable[[], bool], BaseSchedulerNode, BaseSchedulerNode],
        ] = {}

        def fuse_two_nodes(
            node1: BaseSchedulerNode, node2: BaseSchedulerNode
        ) -> BaseSchedulerNode:
            fusion_log.debug("fusing %s with %s", node1.get_name(), node2.get_name())

            device = node1.get_device()
            assert node2.get_device() == device
            node3 = self.get_backend(device).fuse(node1, node2)
            fused_nodes.remove(node1)
            fused_nodes.remove(node2)
            fused_nodes.add(node3)
            self.name_to_fused_node.update(
                {n.get_name(): node3 for n in node3.get_nodes()}
            )
            return node3

        def resolve_pending_fusions(
            node1: BaseSchedulerNode, node2: BaseSchedulerNode
        ) -> None:
            while (
                self.get_fused_node(node1) in pending_fusions
                or self.get_fused_node(node2) in pending_fusions
            ):
                pending_fusion = pending_fusions.get(
                    self.get_fused_node(node1),
                    pending_fusions.get(self.get_fused_node(node2), None),
                )
                assert pending_fusion is not None

                is_speedup, node_key1, node_key2 = pending_fusion
                pending_fusions.pop(node_key1, None)
                pending_fusions.pop(node_key2, None)

                assert self.get_fused_node(node_key1) is node_key1
                assert self.get_fused_node(node_key2) is node_key2

                if not is_speedup() or self.will_fusion_create_cycle(node1, node2):
                    continue

                fuse_two_nodes(node_key1, node_key2)

        for node1, node2 in self.get_possible_fusions(nodes):
            # if either node is in a pending fusion, resolve it.
            # since we iterate on potential fusions based on profitability
            # the first potential fusion should take precedence.
            resolve_pending_fusions(node1, node2)
            node1 = self.get_fused_node(node1)
            node2 = self.get_fused_node(node2)

            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                speedup = self.speedup_by_fusion(node1, node2)
                if callable(speedup):
                    pending_fusions[node1] = (speedup, node1, node2)
                    pending_fusions[node2] = (speedup, node1, node2)
                    continue

                if not speedup:
                    continue

                fuse_two_nodes(node1, node2)

        seen_pair_speedup_fn: OrderedSet[Callable[[], bool]] = OrderedSet()
        for is_speedup_fn, node_key1, node_key2 in pending_fusions.values():
            if is_speedup_fn in seen_pair_speedup_fn:
                continue

            seen_pair_speedup_fn.add(is_speedup_fn)

            assert self.get_fused_node(node_key1) is node_key1
            assert self.get_fused_node(node_key2) is node_key2

            if is_speedup_fn() and not self.will_fusion_create_cycle(
                node_key1, node_key2
            ):
                fuse_two_nodes(node_key1, node_key2)

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
        log.debug("ComboKernels: Generating with num_ck_nodes = %s...", num_ck_nodes)
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
            "Generated ComboKernel nodes: %d ComboKernels, totally %d -> %d nodes",
            count,
            num_nodes_orig,
            len(self.nodes),
        )
        self.prune_redundant_deps(self.nodes)

    def prune_redundant_deps(self, nodes: list[BaseSchedulerNode]) -> None:
        for node in nodes:
            node.prune_redundant_deps(self.name_to_fused_node)

    def get_possible_fusions(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen = OrderedSet[tuple[BaseSchedulerNode, BaseSchedulerNode]]()

        def check_all_pairs(nodes: list[BaseSchedulerNode]) -> None:
            for node1_index, node1 in enumerate(nodes):
                for node2 in nodes[
                    node1_index + 1 : node1_index
                    + 1
                    + config.max_fusion_buffer_group_pairwise_attempts
                ]:
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
        visited = OrderedSet[FusedSchedulerNode]()

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
        Return true if fusing the two nodes can potentially increasing peak memory.

        The implementation is more like a heuristic since we don't really know if we are at peak
        or not when trying to fuse these two nodes. The order of nodes may change later which makes the
        peak memory estimation hard.

        Here is how we decide the LOWER BOUND of extra memory allocation if we fuse these 2 nodes:
        1. find all buffers read by each node with a single user. These buffers are supposed to
           be reused if we don't fuses these 2 nodes
        2. find the intersection of these buffers for the two node and sum the total buffer size.
           If we don't fuse these two nodes, we can at lease avoid this much memory allocation.
           Note that the extra memory allocation is not necessarily causing peak memory increase.
           This is just a heuristic.

        We return true only if the saving for fusion can not trade off the extra memory allocation.
        """

        from .codegen.wrapper import buffer_reuse_key

        def _find_single_user_inputs(
            node: BaseSchedulerNode,
        ) -> list[ir.Buffer]:
            output = []
            for rd in node.read_writes.reads:
                buf = self.name_to_buf.get(rd.name)
                if buf and len(buf.users) == 1 and buf.node.has_tensor_output():
                    output.append(buf.node)
            return output

        # Check inputs that can be potentially reused
        lhs_dep_nodes = _find_single_user_inputs(node1)
        rhs_dep_nodes = _find_single_user_inputs(node2)

        lhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in lhs_dep_nodes)
        rhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in rhs_dep_nodes)

        common_reuse_keys = lhs_reuse_keys.intersection(rhs_reuse_keys)

        memory_overhead = 0
        for key in common_reuse_keys:
            try:
                memory_overhead += int(key[2])
            except ValueError:
                # not an integer. Fallback is to fuse
                return False

        bw_saving = self.score_fusion_memory(node1, node2)

        # The factor 32 here is quite arbitrary.
        if V.graph.sizevars.statically_known_gt(memory_overhead, 32 * bw_saving):
            return True
        return False

    def are_long_distant_nodes(
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
        common_buf_names: Union[tuple[str], OrderedSet[str]],
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

            if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
                reasons[buf_name] = (
                    f"not MemoryDep: {type(lhs_dep)} v.s. {type(rhs_dep)}"
                )
                continue

            if lhs_dep.get_numel() != rhs_dep.get_numel():
                reasons[buf_name] = (
                    f"different numel: {lhs_dep.get_numel()} v.s. {rhs_dep.get_numel()}"
                )
                continue

            # same numel but different MemoryDep.size. Should be broadcasting
            if sympy_product(lhs_dep.size) != sympy_product(rhs_dep.size):
                reasons[buf_name] = "broadcast"
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
            layout_str = ""
            if not isinstance(buf, ir.TorchBindObject):
                layout_str = f"Layout: {buf.layout}"
            reasons[buf_name] = (
                f"Unknown reason: {lhs_dep} v.s. {rhs_dep}. {layout_str}"
            )

        return str(reasons)

    def shared_data_after_reordering_loop(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        """
        Right now just greedily reorder the loop of node1 to be compatible with node2,
        but ideally we should have some heuristics to reorder the loop for node2
        to be compatible with node1 if that's more efficient.
        """

        # TODO Don't do loop reordering for CPU for now.
        # Should debug more why it does not work for CPU codegen
        if not config.loop_ordering_after_fusion or any(
            n.is_cpu() for n in [node1, node2]
        ):
            return 0

        node1_buffer_names = node1.read_writes.buffer_names()
        node2_buffer_names = node2.read_writes.buffer_names()
        # Fast path: no common buffers.
        common_buffer_names = node1_buffer_names & node2_buffer_names
        if not common_buffer_names:
            return 0

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
            return 0

        # Pick the largest buffer to guide the loop reordering
        _numel, lhs_dep, rhs_dep = max(candidates, key=operator.itemgetter(0))

        if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
            return 0

        if lhs_dep.num_vars != rhs_dep.num_vars:
            # this can happen due to we don't merge loops.
            # We can not do loop reordering in this case right now
            # Simply returning true if the two Deps are the same after
            # normalization (merging loops)
            if lhs_dep.normalize() == rhs_dep.normalize():
                return self.dep_size_hint(lhs_dep)
            return 0

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

        return self.score_fusion_memory(node1, node2)

    def unfusable_node(self, node: BaseSchedulerNode) -> bool:
        """
        Is this node unfusable under any conditions.
        """
        return (
            isinstance(node, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node.is_template()
            and not is_output_of_multi_outputs_template(node.node)
        )

    def check_prologue_fusion_heuristics_fusable(
        self,
        prologue_node: BaseSchedulerNode,
        template_node: BaseSchedulerNode,
        why: WhyNoFuse,
    ) -> bool:
        """
        Heuristics to avoid benchmarking predictably slow prologue fusions
        """
        # user opt into more aggressive prologue fusion, dont use heuristics
        if prologue_node.get_operation_names() <= V.graph.invoke_quant_ops:
            return True

        read_bytes = prologue_node.get_read_buffer_sizes()
        write_bytes = prologue_node.get_write_buffer_sizes()

        # Initially, only do fusions which will result in fewer memory accesses inside of the template to avoid
        # potential bad cache behavior and shared memory use.
        # we also want to avoid benchmarking reliably unprofitable fusions like downcasts from fp32 -> fp16 inside kernel.
        # allowing gathers by allowing increasing write_bytes by small factor
        # TODO - make configurable per input, for instance, bias can fuse fp32 -> fp16 profitably

        BYTES_THRESHOLD_MULTIPLIER = 1.1
        if read_bytes > (write_bytes * BYTES_THRESHOLD_MULTIPLIER):
            why("prologue fusion will not increase amount of bytes read in kernel")
            return False

        # we want to avoid attempting to fuse predictably unprofitable prologues
        # such as increasing the unaligned reads or writes.
        # TODO - would be nice to generalize this, however, we would need more explicit
        # knowledge of memory access patterns in the TritonTemplate in order to know
        # the stride order to check alignment.
        origins = tuple(
            e.target
            for n in prologue_node.get_nodes()
            if n.node is not None
            for e in n.node.get_origins()
            if e.op == "call_function"
        )
        if origins == (torch.ops.aten.constant_pad_nd.default,):
            why(
                "prologue fusion will not increase attempt to fuse in padding bc it increases unaligned reads"
            )
            return False

        def low_prec_fp(dtype: torch.dtype) -> bool:
            return dtype.itemsize <= 2 and dtype.is_floating_point

        if (
            low_prec_fp(template_node.get_template_node_or_throw().dtype)
            and not prologue_node.can_codegen_in_low_precision()
        ):
            why(
                "prologue fusion that must be upcast to fp32 not profitable for low precision templates"
            )
            return False

        return True

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """

        if node1 is node2:
            return False

        why = WhyNoFuse(node1, node2)

        if node1.is_template() and self.get_backend(
            node1.get_device()
        ).can_fuse_multi_outputs_template(node1, node2):
            return True

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
            if not config.prologue_fusion:
                why("prologue fusion turned off")
                return False

            if node1.is_reduction() or node1.is_template():
                why("prologue fusion only supported for pointwise nodes")
                return False

            template = node2.get_template_node_or_throw()
            if not isinstance(template, ir.TritonTemplateBuffer):
                why("prologue fusion only supported for TritonTemplates")
                return False

            allowed_prologue_inps = template.get_allowed_prologue_inps()

            unsupported_prologue_args = (
                OrderedSet(inp.get_name() for inp in template.inputs)
                - allowed_prologue_inps
            )

            if node1.get_buffer_names() & unsupported_prologue_args:
                why("prologue fusion not implemented for kernel for these inputs")
                return False

            if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
                why("template prologue can only fuse functional pointwise nodes")
                return False

            prologue_nodes = node1.get_nodes()
            for node in prologue_nodes[:-1]:
                node_outs = node.get_outputs()
                for out in node_outs:
                    if not all(user.node in prologue_nodes for user in out.users):
                        why("template prologue can only fuse nodes with a single use")
                        return False

            template_snodes = (
                [node2]
                if not isinstance(node2, FusedSchedulerNode)
                else [n for n in node2.snodes if n.is_template()]
            )
            assert len(template_snodes) == 1
            template_snode = template_snodes[0]

            if not (
                len(prologue_nodes[-1].outputs) == 1
                and len(prologue_nodes[-1].outputs[0].users) == 1
                and prologue_nodes[-1].outputs[0].users[0].node is template_snode
            ):
                why(
                    "template prologue can only fuse nodes with a single use into template"
                )
                return False

            if not self.check_prologue_fusion_heuristics_fusable(node1, node2, why):
                return False

        if node1.is_template() and (
            node2.has_aliasing_or_mutation()
            or node2.is_reduction()
            or not config.epilogue_fusion
        ):
            why("template epilogue not satisfied")
            return False

        if (node1.get_buffer_names() & V.graph.no_fuse_buffer_names) or (
            node2.get_buffer_names() & V.graph.no_fuse_buffer_names
        ):
            why("fusion for buffer explicit disabled")
            return False

        device = node1.get_device()
        device2 = node2.get_device()
        if device != device2:
            why("device mismatch (%s vs %s)", device, device2)
            return False
        del device2

        shared_data_score = self.score_fusion_memory(node1, node2)
        if (
            shared_data_score < config.score_fusion_memory_threshold
            and config.loop_ordering_after_fusion
        ):
            shared_data_score = self.shared_data_after_reordering_loop(node1, node2)

        if loop_ordering_log.isEnabledFor(logging.DEBUG):
            loop_ordering_log.debug(
                "%s and %s has %s shared data",
                node1.get_name(),
                node2.get_name(),
                shared_data_score,
            )

        if not V.choices.can_fuse(self, node1, node2, shared_data_score):
            return False

        if node1.get_operation_names() & node2.ancestors:
            # node2 depends on node1 outputs
            return (
                self.can_fuse_vertical(node1, node2)
                and V.choices.can_fuse_vertical(self, node1, node2, shared_data_score)
                and self.get_backend(device).can_fuse_vertical(node1, node2)
            )
        else:  # nodes don't depend on each other, but may have common reads
            return V.choices.can_fuse_horizontal(
                self, node1, node2, shared_data_score
            ) and self.get_backend(device).can_fuse_horizontal(node1, node2)

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
        remaining_deps_by_name: dict[str, list[Dep]] = defaultdict(list)

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
            dep.name
            for dep in itertools.chain.from_iterable(remaining_deps_by_name.values())
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
            op_name = self.name_to_buf[name].defining_op_name()
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
        if min(node1_dep_len, node2_dep_len) * 4 < max(node1_dep_len, node2_dep_len):
            if node1_dep_len > node2_dep_len:
                tmp = node1
                node1 = node2
                node2 = tmp

            deps = [
                dep
                for dep in node1.read_writes.reads | node1.read_writes.writes
                if dep in node2.read_writes.reads or dep in node2.read_writes.writes
            ]

            return sum(self.dep_size_hint(dep) for dep in deps)

        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        return sum(self.dep_size_hint(dep) for dep in common_memory_deps)

    def get_possible_fusions_with_highest_priority(
        self, possible_fusions: list[tuple[BaseSchedulerNode, BaseSchedulerNode]]
    ) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        # Group the possible fusions based on their priority from the backend.
        # Only return the group of possible fusions with highest priority.
        if len(possible_fusions) == 0:
            return possible_fusions
        possible_fusions_group_by_priority: dict[
            int, list[tuple[BaseSchedulerNode, BaseSchedulerNode]]
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
        self, nodes: tuple[BaseSchedulerNode, BaseSchedulerNode]
    ) -> Any:
        """
        Shim for list.sort(key=...)
        """
        return V.choices.score_fusion(self, *nodes)

    def compute_last_usage(self) -> None:
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

        future_used_buffers = OrderedSet(V.graph.get_output_names())

        for node in reversed(self.nodes):
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)

    def free_buffers(self) -> None:
        """Free any buffers that are no longer needed"""
        for name in sorted(
            self.buffer_names_to_free
            - V.graph.removed_buffers
            - V.graph.wrapper_code.freed  # type: ignore[has-type]
        ):
            if name in self.name_to_buf:
                buf = self.name_to_buf[name]
                if buf.can_free():
                    V.graph.wrapper_code.codegen_free(buf.node)
            elif name in V.graph.graph_inputs:
                inp = V.graph.graph_inputs[name]
                if isinstance(inp, ir.TorchBindObject):
                    V.graph.wrapper_code.codegen_free(inp)
                elif isinstance(inp, ir.GeneratorState):
                    continue
                else:
                    storage = inp.data
                    assert (
                        isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
                    )
                    V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

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
        assert not is_gpu(device.type) or device.index is not None, (
            f"{device} should have been normalized in lowering"
        )
        V.graph.add_device_info(device)

        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise GPUTooOldForTriton(device_props, inspect.currentframe())
            elif is_gpu(device.type) and not device.type == "mps":
                raise TritonMissing(inspect.currentframe())

        return device_scheduling(self)

    def get_backend(self, device: Optional[torch.device]) -> BaseScheduling:
        assert device is not None
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

    def can_buffer_be_removed_through_fusion(
        self, name: str, fused_node_names: OrderedSet[str]
    ) -> bool:
        try:
            users = self.name_to_buf[name].users
        except KeyError:
            return False
        return (
            all(user.is_weak or user.get_name() in fused_node_names for user in users)
            and name not in self.mutation_renames
            and name not in self.mutation_real_name
        )

    def should_partition(self, node: BaseSchedulerNode) -> bool:
        """Return True if we should partition the inductor graph on this node"""
        if isinstance(node, FusedSchedulerNode):
            return any(self.should_partition(snode) for snode in node.snodes)

        if not node.is_gpu():
            return True

        if node.node is None:
            return True

        if isinstance(node.node, ir.DeviceCopy):
            return True

        if isinstance(node.node, ir.Conditional):
            return True

        if getattr(node.node, "unbacked_bindings", None):
            return True

        if is_cudagraph_unsafe_op(node.node):
            return True

        return False

    def get_name_to_nodes(
        self,
    ) -> dict[str, Union[ir.IRNode, ir.TorchBindObject, sympy.Expr]]:
        """
        Return a mapping from name strings to the corresponding graph inputs or
        base scheduler node outputs.
        """
        name_to_node: dict[str, Union[ir.IRNode, ir.TorchBindObject, sympy.Expr]] = {}
        name_to_node.update(V.graph.graph_inputs)

        for node in self.nodes:
            for name, scheduler_buffer in node.outputs_by_name.items():
                name_to_node[name] = scheduler_buffer.node

        return name_to_node

    def compute_graph_partition_maps(
        self,
        signatures: list[GraphPartitionSignature],
    ) -> None:
        """
        computes a mapping from partition input/output indices to graph input/output
        indices for each partition.
        """
        name_to_graph_input_index = {
            name: idx for idx, name in enumerate(V.graph.graph_inputs)
        }
        name_to_graph_output_index = {
            name: idx for idx, name in enumerate(V.graph.get_output_names())
        }

        V.graph.partition_maps = []
        for partition_id, signature in enumerate(signatures):
            if signature.skip_cudagraph:
                # Note: [Graph Partition Map for CUDAGraph]
                # number of partition map should be the same as the number of generated
                # partition functions. This assumption will be used when cudagraphify
                # each partition function.
                continue

            input_mapping = []
            for name in signature.input_nodes:
                input_mapping.append(name_to_graph_input_index.get(name))

            output_mapping = []
            for node in signature.output_nodes:
                output_mapping.append(name_to_graph_output_index.get(node.get_name()))

            V.graph.partition_maps.append(
                GraphPartitionMap(
                    partition_id,
                    input_mapping,
                    output_mapping,
                    signature.constant_names,
                )
            )

    def get_graph_partition_symbol_inputs(
        self,
        partition: PartitionType,
        input_nodes: dict[str, Union[ir.IRNode, ir.TorchBindObject, sympy.Expr]],
    ) -> OrderedSet[sympy.Symbol]:
        """
        Returns all symbol inputs which are required to be in scope to successfully
        perform codegen for this graph partition, including:
        - free symbols used in partition nodes
        - free symbols in partition input/node shapes, strides, and offsets. This is needed
          for recording cudagraphs for tensors with dynamic shapes.
        """

        def get_layout_symints(node: ir.IRNode) -> OrderedSet[sympy.Symbol]:
            free_symbol_uses: OrderedSet[sympy.Symbol] = OrderedSet()
            layout = node.maybe_get_layout()
            if isinstance(layout, ir.Layout):
                free_symbol_uses.update(
                    free_symbols(layout.size)
                    | free_symbols(layout.stride)
                    | free_symbols(layout.offset)
                )
                if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
                    # symint may be used as index in layout.target
                    free_symbol_uses.update(get_layout_symints(layout.target))
            else:
                assert layout is None, (
                    f"Expect layout to be None but found layout={layout}"
                )
            return free_symbol_uses

        def get_scheduler_node_symbol_uses(
            node: BaseSchedulerNode,
        ) -> OrderedSet[sympy.Symbol]:
            """
            Gets symbols used in node.
            """
            if isinstance(node, FusedSchedulerNode):
                return OrderedSet().union(
                    *(get_scheduler_node_symbol_uses(snode) for snode in node.snodes)
                )
            assert node.node is not None
            free_symbol_uses = node.node.get_free_symbol_uses()
            free_symbol_uses.update(
                *(get_layout_symints(ir_node) for ir_node in node.node.get_outputs())
            )
            return free_symbol_uses

        def get_input_node_symbols(
            node: Union[ir.IRNode, sympy.Expr, ir.TorchBindObject],
        ) -> OrderedSet[sympy.Symbol]:
            """
            Gets symbols used in input node shapes, strides, and offsets.
            """
            if isinstance(node, ir.TorchBindObject):
                # TorchBindObject does not involve dynamic shapes yet
                return OrderedSet()
            elif isinstance(node, ir.IRNode):
                return get_layout_symints(node)
            else:
                # node cannot be sympy.Expr since node comes from read_writes and
                # read_writes does not contain sympy.Expr
                raise NotImplementedError(f"Unsupported input node type: {type(node)}")

        def filter_symbols(
            symbols: OrderedSet[sympy.Symbol],
        ) -> OrderedSet[sympy.Symbol]:
            """
            Filters a set of symbols that are required for codegen. Skip symbols
            that are always internal to kernels, such as SymT.TMP, SymT.INDEX,
            and SymT.R0_INDEX.
            """
            return OrderedSet(
                s
                for s in symbols
                if symbol_is_type(
                    s,
                    (
                        SymT.SIZE,
                        SymT.FLOAT,
                        SymT.UNBACKED_INT,
                        SymT.UNBACKED_FLOAT,
                    ),
                )
            )

        candidate_symbols: OrderedSet[sympy.Symbol] = OrderedSet().union(
            *(get_scheduler_node_symbol_uses(node) for node in partition)
        )
        candidate_symbols.union(
            *(get_input_node_symbols(node) for _, node in input_nodes.items())
        )

        candidate_symbols = filter_symbols(candidate_symbols)

        res: OrderedSet[sympy.Symbol] = OrderedSet()
        for s in candidate_symbols:
            symplified_s = V.graph.sizevars.simplify(s)
            # use free_symbols only when s is simplified to an Integer or expr
            res.update(symplified_s.free_symbols)

        return OrderedSet(sorted(res, key=operator.attrgetter("name")))

    def get_graph_partition_signature(
        self, partitions: list[PartitionType], skip_cudagraphs: list[bool]
    ) -> list[GraphPartitionSignature]:
        """
        Gets signature for each graph partition, including input nodes, output nodes, and
        whether deallocating an input within graph partition.
        """
        signatures = []

        unmet_output_names = OrderedSet(V.graph.get_output_names())
        name_to_node = self.get_name_to_nodes()

        def is_none_layout(buf_name: str) -> bool:
            """
            Checks if buf_name is NoneLayout. Buffers with NoneLayout is not allocated
            so graph partition should not take it as inputs or outputs.
            """
            buf = self.name_to_buf.get(buf_name, None)

            if buf is None:
                return False

            if isinstance(buf.node.layout, NoneLayout):
                if isinstance(buf.node, ir.MutationOutput) and (
                    real_name := self.mutation_real_name.get(buf_name, None)
                ):
                    return is_none_layout(real_name)

                return True

            return False

        for partition, skip_cudagraph in zip(
            reversed(partitions), reversed(skip_cudagraphs)
        ):
            output_names: OrderedSet[str] = OrderedSet()

            for node in partition:
                output_names.update(node.outputs_by_name.keys())

            returned_output_names = output_names.intersection(unmet_output_names)

            # all reads/writes are partition inputs except those generated
            # within the partition and tensor constants
            read_writes = dependencies.ReadWrites.merge_list(
                [node.read_writes for node in partition]
            )

            # WeakDep is fake dependency on unused buffer. It should not appear
            # in partition_input_names for inputs that are actually read or written.
            partition_input_names = (
                OrderedSet(
                    [
                        x.name
                        for x in read_writes.reads | read_writes.writes
                        if not is_none_layout(x.name)
                    ]
                )
                - output_names
            )

            partition_input_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in partition_input_names
            )

            buffer_names_to_free: OrderedSet[str] = OrderedSet()
            for node in partition:
                buffer_names_to_free.update(node.last_usage)

            input_nodes = {
                name: name_to_node[name]
                for name in partition_input_names
                if name in name_to_node
            }
            input_deallocation = {
                name: True if name in buffer_names_to_free else False
                for name in partition_input_names
                if name in name_to_node
            }

            # if an input tensor is not freed in the partition function, it should
            # also be returned as an output. This brings benefits to cudagraph
            # since the returned output tensor is a cudagraph managed tensor with
            # a static tensor address.
            extra_output_names = [
                name
                for name in partition_input_names
                if name in name_to_node and name not in buffer_names_to_free
            ]

            returned_output_names.update(extra_output_names)

            returned_output_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in returned_output_names
            )

            output_nodes = [
                name_to_node[name]
                for name in returned_output_names
                if not is_none_layout(name)
            ]

            constant_names = [
                name for name in partition_input_names if name in V.graph.constants
            ]

            symbol_inputs = self.get_graph_partition_symbol_inputs(
                partition, input_nodes
            )

            partition_signature = GraphPartitionSignature(
                symbol_inputs,
                input_nodes,
                output_nodes,
                input_deallocation,
                skip_cudagraph,
                constant_names,
            )

            signatures.append(partition_signature)

            unmet_output_names = partition_input_names.union(
                unmet_output_names - returned_output_names
            )

        return signatures[::-1]

    def clean_removed_buffer_from_partition_signatures(
        self, signature: GraphPartitionSignature
    ) -> GraphPartitionSignature:
        """
        Updates the partition signature by removing buffers specified in
        V.graph.removed_buffers. See [Note: Removed Graph Partition Arguments]
        """
        input_nodes = {
            name: buffer
            for name, buffer in signature.input_nodes.items()
            if name not in V.graph.removed_buffers
        }
        input_deallocation = {
            name: val
            for name, val in signature.input_deallocation.items()
            if name not in V.graph.removed_buffers
        }
        output_nodes = [
            node
            for node in signature.output_nodes
            if node.maybe_get_name() not in V.graph.removed_buffers
        ]
        constant_names = [
            name
            for name in signature.constant_names
            if name not in V.graph.removed_buffers
        ]
        return GraphPartitionSignature(
            signature.symbol_inputs,
            input_nodes,
            output_nodes,
            input_deallocation,
            signature.skip_cudagraph,
            constant_names,
        )

    def reorder_for_minimizing_partition(
        self,
        nodes: list[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        """
        Reorder nodes to minimize the number of partitions via a bfs
        topological sort. This is the optimal reordering such that the
        number of partitions cannot be reduced further. This may be
        sub-optimal for other metrics such as peak memory. This does not
        change relative orders of two cudagraphable nodes, nor the
        relative order of two non_cudagraphable nodes.
        """
        import heapq

        node_to_indegree: dict[BaseSchedulerNode, int] = dict()
        cudagraphable_nodes: list[tuple[int, BaseSchedulerNode]] = []
        non_cudagraphable_nodes: list[tuple[int, BaseSchedulerNode]] = []
        node_to_index = {node: idx for idx, node in enumerate(nodes)}

        def insert_pending_nodes(node: BaseSchedulerNode) -> None:
            node_with_index = (node_to_index[node], node)
            if self.should_partition(node):
                heapq.heappush(non_cudagraphable_nodes, node_with_index)
            else:
                heapq.heappush(cudagraphable_nodes, node_with_index)

        def update_indegree(node: BaseSchedulerNode) -> None:
            for succ_node in node.mpi_node.succ_nodes:
                assert node_to_indegree[succ_node] > 0
                node_to_indegree[succ_node] -= 1
                if node_to_indegree[succ_node] == 0:
                    insert_pending_nodes(succ_node)

        for node in nodes:
            node_to_indegree[node] = len(node.mpi_node.pred_nodes)
            if node_to_indegree[node] == 0:
                insert_pending_nodes(node)

        schedule: list[BaseSchedulerNode] = []
        num_iters: int = 0
        while num_iters < len(nodes) and (
            non_cudagraphable_nodes or cudagraphable_nodes
        ):
            while non_cudagraphable_nodes:
                _, node = heapq.heappop(non_cudagraphable_nodes)
                schedule.append(node)
                update_indegree(node)

            while cudagraphable_nodes:
                _, node = heapq.heappop(cudagraphable_nodes)
                schedule.append(node)
                update_indegree(node)

            num_iters += 1

        if num_iters > len(nodes):
            raise RuntimeError(
                """
                Failed to schedule, while loop ran too long when
                reordering for minimizing the num of partitions
                """
            )

        return schedule

    def maybe_reorder_for_minimizing_partition(
        self,
        nodes: list[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        """
        Reorder nodes to minimize the number of partitions if this only slightly
        increase peak memory.
        """
        from .memory import estimate_peak_memory, prepare_planning_info

        graph_outputs = OrderedSet(V.graph.get_output_names())

        default_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
            nodes,
            self.name_to_buf,
            self.name_to_fused_node,
            OrderedSet(V.graph.graph_inputs.keys()),
            graph_outputs,
        )

        reordered_nodes = self.reorder_for_minimizing_partition(nodes)
        reorder_peak_memory, _ = estimate_peak_memory(
            reordered_nodes, name_to_freeable_input_buf, graph_outputs
        )

        # 1.1 here means 10% extra peak memory budget which is quite arbitrary
        if reorder_peak_memory < default_peak_memory * 1.1:
            return reordered_nodes

        return nodes

    def reorder_for_partition_with_simple_dependency(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        """
        Reorder a node if it should be partitioned and has simple dependency:
        1. move a partitioned node to the front if it has no dependency
        2. move a partitioned node to the back if it is only used by OutputNode
        3. otherwise do not reorder
        """

        front: list[BaseSchedulerNode] = []
        middle: list[BaseSchedulerNode] = []
        back: list[BaseSchedulerNode] = []

        def only_output_user(node: BaseSchedulerNode) -> bool:
            for buf in node.get_outputs():
                for use in buf.users:
                    if not isinstance(use.node, OutputNode):
                        return False
            return True

        for node in nodes:
            should_partition = self.should_partition(node)
            if should_partition and len(node.unmet_dependencies) == 0:
                front.append(node)
            elif should_partition and only_output_user(node):
                back.append(node)
            else:
                middle.append(node)

        return front + middle + back

    def graph_partition(
        self,
    ) -> tuple[list[PartitionType], list[GraphPartitionSignature]]:
        """
        Given a list of BaseSchedulerNodes, split into a list of
        graph partitions and compute partition input/output signatures.
        """
        partitions: list[PartitionType] = []
        skip_cudagraph = True
        cur_partition: PartitionType = []
        skip_cudagraphs = []
        for node in self.nodes:
            should_partition = self.should_partition(node)
            if cur_partition and skip_cudagraph != should_partition:
                partitions.append(cur_partition)
                skip_cudagraphs.append(skip_cudagraph)
                cur_partition = []

            skip_cudagraph = should_partition
            cur_partition.append(node)

        if cur_partition:
            partitions.append(cur_partition)
            skip_cudagraphs.append(skip_cudagraph)

        signatures = self.get_graph_partition_signature(
            partitions=partitions, skip_cudagraphs=skip_cudagraphs
        )
        self.compute_graph_partition_maps(signatures)

        return partitions, signatures

    def codegen(self) -> None:
        with dynamo_timed("Scheduler.codegen"):
            return (
                self._codegen_partitions()
                if torch._inductor.config.graph_partition
                else self._codegen(self.nodes)
            )

    def _codegen_partition_wrapper(
        self,
        partition: PartitionType,
        signature: GraphPartitionSignature,
    ) -> None:
        """Codegen a partition given its inputs/outputs"""
        from .codegen.wrapper import SubgraphPythonWrapperCodegen

        parent_wrapper_code = V.graph.wrapper_code
        graph_partition_id = next(self._graph_partition_counter)

        with V.graph.set_current_wrapper_code():
            V.graph.init_wrapper_code(
                is_subgraph=True,
                subgraph_name=f"partition_{graph_partition_id}",
                parent_wrapper_code=parent_wrapper_code,
                partition_signatures=signature,
            )
            self._codegen(partition)

            # Note: [Removed Graph Partition Arguments]
            # Graph partition relies on node.read_writes to analyze the partition
            # inputs and outputs. However, during codegen, we may decide some buffers
            # are internal to a kernel (e.g., triton kernel) such that these buffers
            # are never actually defined. This information is collected during codegen
            # and recorded in V.graph.removed_buffers. So we cleanup signature and write
            # prefix (i.e., generating call function and return outputs) after we have
            # codegen the partition.
            assert isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen)
            signature = self.clean_removed_buffer_from_partition_signatures(signature)
            V.graph.wrapper_code.partition_signatures = signature
            V.graph.wrapper_code.write_prefix()

            partition_code, _ = V.graph.wrapper_code.generate(V.graph.is_inference)

        V.graph.wrapper_code.define_subgraph_launcher_fn(partition_code.value)

        V.graph.wrapper_code.codegen_partition_call(graph_partition_id, signature)
        V.graph.wrapper_code.allocated.update(  # type: ignore[has-type]
            [node.get_name() for node in signature.output_nodes]
        )

    def _codegen_partitions(self) -> None:
        """
        Split nodes into partitions and codegen each partition into separate functions.
        This allows further applying different optimizations (e.g., cudagraph) to
        each function.
        """
        partitions, signatures = self.graph_partition()

        for partition, signature in zip(partitions, signatures):
            assert len(partition) >= 1, (
                f"Each partition must have at least one node but found {len(partition)}"
            )

            if signature.skip_cudagraph:
                self._codegen(partition)
            else:
                self._codegen_partition_wrapper(partition, signature)

        num_partitions = next(self._graph_partition_counter)
        V.graph.wrapper_code.set_all_partition_names(num_partitions)

        # See [Note: Graph Partition Map for CUDAGraph]
        if num_partitions > 0:
            assert V.graph.partition_maps is not None
            assert num_partitions == len(V.graph.partition_maps), (
                f"Expect {num_partitions} partition maps but got {len(V.graph.partition_maps)}"
            )

    def _codegen(self, nodes: list[BaseSchedulerNode]) -> None:
        if config.check_stack_no_cycles_TESTING_ONLY:
            import torch._dynamo.convert_frame

            stack = traceback.extract_stack()
            seen: OrderedSet[tuple[str, int | None]] = OrderedSet()
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
        for node in nodes:
            if log.isEnabledFor(logging.DEBUG):
                try:
                    log.debug(
                        "Generating code for node %s with estimated runtime %f",
                        node.get_name(),
                        node.get_estimated_runtime(),
                    )
                except Exception:
                    log.debug(
                        "Generating code for node %s with estimated runtime 0.0",
                        node.get_name(),
                    )

            self.enter_context(node)

            if device := node.get_device():
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
                prologue, template_node, epilogue = node.get_prologue_template_epilogue(
                    list(node.get_nodes())
                )
                self.get_backend(device).codegen_template(
                    template_node, epilogue, prologue
                )
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
                if (
                    device is not None
                    and device.type != "meta"
                    and self.get_backend(device).ready_to_flush()
                ):
                    self.flush()

        if self.current_device and device_need_guard(self.current_device.type):
            # exit the outermost CUDA device guard. this is
            # important for nested indentation codegen-ing.
            V.graph.wrapper_code.codegen_device_guard_exit()

        self.flush()

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> tuple[float, float, list[Optional[str]]]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        device = node_list[0].get_device()
        V.graph.scheduler = self
        self.current_device = device
        assert device is not None
        backend = self.get_backend(device)
        return backend.benchmark_combo_kernel(node_list)

    def speedup_by_combo_kernel(self, nodes: list[BaseSchedulerNode]) -> bool:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """
        if not config.benchmark_combo_kernel:
            return True

        subkernel_nodes = nodes
        device = subkernel_nodes[0].get_device()

        # don't support benchmark fusion for CPU right now.
        if device is None or device.type == "cpu":
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
                # workaround triton issue: https://github.com/triton-lang/triton/issues/2151
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
            ms2, ms2_clone, _path2_list = self.benchmark_combo_kernel(subkernel_nodes)
        except CompilationError as e:
            # workaround triton issue: https://github.com/triton-lang/triton/issues/2151
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
            if node.is_gpu():
                for read in node.read_writes.reads:
                    buffer = V.graph.name_to_buffer.get(read.name)
                    if (
                        buffer
                        and get_device_type(buffer) == "cpu"
                        and not isinstance(
                            buffer.layout, (NoneLayout, MultiOutputLayout)
                        )
                        and buffer.get_size() == []
                    ):
                        V.graph.zero_dim_cpu_tensor_list.add(read.name)


class BaseScheduling:
    def __init__(self, scheduler: Optional[Scheduler]):
        super().__init__()
        self.scheduler = scheduler

    def free_buffers_in_scheduler(self) -> None:
        if self.scheduler:
            self.scheduler.free_buffers()

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        """Return a set of .codegen.common.BackendFeature()"""
        return OrderedSet()

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

    def can_fuse_multi_outputs_template(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        A Multi-Output Template (referenced in #144012) is a template node
        with MultiOutputLayout, and its output buffers are instances of MultiOutput.
        In this context, we verify whether node1 represents the Multi-Output Template
        and node2 corresponds to one of its outputs. If so, we further check if
        backend supports this fusion.
        """
        return False

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
    ) -> tuple[tuple[sympy.Expr, ...], ...]:
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError

    def generate_kernel_code_from_nodes(
        self, nodes: Sequence[BaseSchedulerNode], benchmark_kernel: bool
    ) -> str:
        """
        Generate a kernel given a list of pre-fused nodes.
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
    ) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def benchmark_codegened_module(self, module: ModuleType) -> tuple[float, str]:
        """
        Benchmark a compiled module and return the execution time
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
    ) -> tuple[float, float, list[Optional[str]]]:
        """
        Benchmark the list of nodes to combine and return the execution time
        and memory copy time in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError
