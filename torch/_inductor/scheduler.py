import collections
import dataclasses
import functools
import itertools
import logging
import os
import pprint
import textwrap
from typing import Dict, List, Optional, Set

import sympy

import torch
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton

from . import config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
    cache_on_self,
    cmp,
    free_symbol_has,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    sympy_product,
)
from .virtualized import V


log = logging.getLogger(__name__)


def pformat(obj):
    if isinstance(obj, set):
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' '*4)}"
    return result


class OutputNode:
    def __init__(self, dep):
        self.unmet_dependencies = {dep}
        self.inverse_users = []

    def is_reduction(self):
        return False

    def get_alias_names(self):
        return ()

    def get_name(self):
        return "OUTPUT"

    __repr__ = get_name


def fuse(node1: "BaseSchedulerNode", node2: "BaseSchedulerNode"):
    if node1.is_foreach() or node2.is_foreach():
        return ForeachKernelSchedulerNode.fuse(node1, node2)
    else:
        return FusedSchedulerNode.fuse(node1, node2)


# TODO(xmfan): reuse an existing mapping for this if it exists, or formalize this into ir.py:ExternKernel
kernel_name_to_op = {
    "extern_kernels.convolution": torch.ops.aten.convolution,
    "extern_kernels.mm": torch.ops.aten.mm,
    "extern_kernels.bmm": torch.ops.aten.bmm,
    "extern_kernels.addmm": torch.ops.aten.addmm,
}


class BaseSchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.Buffer):
        self.scheduler: Scheduler = scheduler
        self.node: ir.Buffer = node
        self.users: Optional[List[NodeUser]] = None
        self.inverse_users: List[BaseSchedulerNode] = []
        self.set_read_writes(node.get_read_writes())
        self.ancestors: Optional[Set[str]] = None
        self.min_order: Optional[int] = None
        self.max_order: Optional[int] = None
        self.last_usage: Set[str] = None  # buffers that won't be used after this kernel
        self.written = False

    def __repr__(self):
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        lines = [
            f"{name}: {type(self).__name__}({type(self.node).__name__})",
            f"{name}.writes = {pformat(self.read_writes.writes)}",
            f"{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}",
            f"{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}",
            f"{name}.users = {self.users}",
        ]
        try:
            lines += [
                self.debug_str_extra(),
            ]
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return "\n".join(lines).rstrip()

    def debug_str_extra(self) -> str:
        return ""

    def log_details(self):
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def update_mutated_names(self, renames: Dict[str, str]):
        self.set_read_writes(self.read_writes.rename(renames))

    def add_mutation_dep(self, dep):
        self.set_read_writes(self.read_writes.with_read(dep))

    def set_users(self, users: List["NodeUser"]):
        # deduplicate
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def set_last_usage(
        self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
    ):
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = {mutation_real_name.get(k, k) for k in used_buffers}
        self.last_usage = used_buffers - future_used_buffers

    def get_aliases(self):
        return self.node.get_alias_names()

    def get_mutations(self):
        return self.node.get_mutation_names()

    def has_aliasing_or_mutation(self):
        return bool(self.get_aliases() or self.get_mutations())

    def set_read_writes(self, rw: dependencies.ReadWrites):
        self.read_writes: dependencies.ReadWrites = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def op_counts(self):
        return self.read_writes.op_counts

    def used_buffer_names(self) -> Set[str]:
        return {
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        }

    def used_or_aliased_buffer_names(self) -> Set[str]:
        used_names = set()

        for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes):
            used_names.add(dep.name)
            if V.graph.name_to_buffer.get(dep.name):
                layout = V.graph.name_to_buffer[dep.name].get_layout()
                # needed to avoid deallocating aliased buffer
                # if there are still uses of aliases ahead
                if isinstance(layout, ir.AliasedLayout):
                    used_names.add(layout.view.data.get_name())
        return used_names

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def prune_weak_deps(self):
        # Prune weak dependencies on buffers that have been removed
        def should_prune(dep):
            return isinstance(dep, WeakDep) and dep.name in V.graph.removed_buffers

        to_remove = {dep for dep in self.read_writes.reads if should_prune(dep)}
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(self, name_to_fused_node):
        """
        Prunes stardeps intended for mutation ordering
        on an upstream fused node if after fusion there is another dependency
        on the fused upstream node, making the stardep redundant

        In essence this enforces an ordering on fusions. As fusions occur, prunable stardeps will
        be incrementally removed, enabling other fusions, ensuring they are fused in order.
        """
        name_to_dep_count = collections.Counter()

        for dep in self.unmet_dependencies:
            if not isinstance(dep, WeakDep):
                name_to_dep_count[name_to_fused_node[dep.name].get_name()] += 1

        def should_prune(dep):
            if isinstance(dep, WeakDep):
                is_redundant = (
                    name_to_dep_count[name_to_fused_node[dep.name].get_name()] > 0
                )
                # These can occur because fused nodes always gather deps from their snodes
                # If B has a weakdep on A
                # B gets fused with C, then any time BC is fused, the weakdep will reappear
                is_self_dep = name_to_fused_node[dep.name] == self
                return is_redundant or is_self_dep
            else:
                return False

        deps_to_prune = {dep for dep in self.unmet_dependencies if should_prune(dep)}
        self.unmet_dependencies = self.unmet_dependencies - deps_to_prune
        self.set_read_writes(self.read_writes.remove_reads(deps_to_prune))

    def get_name(self) -> str:
        return self.node.get_name()

    def get_first_name(self) -> str:
        return self.get_name()

    def get_names(self) -> Set[str]:
        return {self.get_name()}

    def get_nodes(self) -> List["BaseSchedulerNode"]:
        return [self]

    def get_device(self):
        return self.node.get_device()

    def is_reduction(self):
        return False

    def is_template(self):
        return False

    def is_extern(self):
        return False

    def is_foreach(self):
        return False

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def has_side_effects(self):
        return False

    def decide_inplace_update(self):
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
        if not self.node.should_allocate():
            return

        if isinstance(self, (SchedulerNode,)) and (
            self.node.get_alias_names() or self.node.get_mutation_names()
        ):
            return

        if (
            (
                isinstance(self, (SchedulerNode,))
                # o what have i done.  lets make this an api
                or (
                    isinstance(self, ExternKernelSchedulerNode)
                    and isinstance(self.node, (ir.AllReduce, ir.InPlaceHint))
                )
            )
            and config.inplace_buffers
            and (
                not isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel)
                or getattr(V.kernel, "mutations", None) is not None
            )
        ):
            from .codegen.wrapper import buffer_reuse_key

            ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)

            for read in ordered_reads:
                input_node: BaseSchedulerNode = self.scheduler.name_to_node.get(
                    read.name
                )
                if input_node and V.graph.wrapper_code.can_reuse(input_node, self):
                    remaining_uses = [
                        x
                        for x in input_node.users
                        if x.node.get_name()
                        not in self.scheduler.available_buffer_names
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and not isinstance(
                            input_node.node.get_layout(),
                            (
                                ir.MultiOutputLayout,
                                ir.MutationLayout,
                                ir.AliasedLayout,
                            ),
                        )
                        and buffer_reuse_key(input_node.node)
                        == buffer_reuse_key(self.node)
                    ):
                        # hacky check for if V.kernel is a real kernel or NullHandler
                        if hasattr(V.kernel, "args"):
                            # if there isn't a triton kernel, then we don't need to call triton-specific things.
                            # but TODO this might be a convenient place to signal to the Collective kernels to inplace
                            # (and, can we make "kernel" less generic of a name?)
                            V.kernel.args.make_inplace(
                                input_node.get_name(), self.get_name()
                            )
                            # mutations not tracked in cpp kernels
                            if isinstance(
                                V.kernel, torch._inductor.codegen.triton.TritonKernel
                            ):
                                V.kernel.mutations.add(input_node.get_name())
                                V.kernel.mutations.add(self.get_name())

                            # update last usage of reused node
                            self.last_usage.discard(input_node.get_name())

                            V.kernel.inplace_update_buffers[
                                self.get_name()
                            ] = input_node.get_name()
                        break

    def allocate(self):
        if not self.node.should_allocate():
            return

        if isinstance(self, (SchedulerNode,)) and (
            self.node.get_alias_names() or self.node.get_mutation_names()
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # hacky check for if V.kernel is a real kernel or NullHandler
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            V.graph.wrapper_code.codegen_inplace_reuse(
                self.scheduler.name_to_node[
                    V.kernel.inplace_update_buffers[self.get_name()]
                ].node,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def codegen_originating_info(self, buffer, only_once=True):
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        origins = self.node.origins
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
            return 0

        if isinstance(self, SchedulerNode):
            node_numel = sympy_product(self.get_ranges()[0]) * sympy_product(
                self.get_ranges()[1]
            )
        else:
            node_numel = int(1e9)
        buf_accesses = collections.defaultdict(list)
        for dep in self.read_writes.reads | self.read_writes.writes:
            buf_accesses[dep.name].append(dep)

        reads = {dep.name for dep in self.read_writes.reads}
        writes = {dep.name for dep in self.read_writes.writes}

        def is_materialized(buf):
            buf_uses = {user.node for user in self.scheduler.name_to_node[buf].users}
            return len(buf_uses - set(self.snodes)) > 0

        if isinstance(self, FusedSchedulerNode):
            removed_buffers = {dep for dep in writes if not is_materialized(dep)}
            writes = writes - removed_buffers
            reads = reads - removed_buffers
        node_bytes = 0

        for buf in reads | writes:
            buf_accessed_elems = sum([node_numel for dep in buf_accesses[buf]])
            if buf in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf]
            elif buf in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf]
            else:
                continue

            def get_buf_elems(buf):
                return V.graph.sizevars.size_hint(sympy_product(buf.get_size()))

            # Kind of a lazy way to get the MultiOutput nodes corresponding to
            # a MultiOutputLayout
            if isinstance(buf.layout, MultiOutputLayout):
                buf_elems = sum(
                    get_buf_elems(user.node.node)
                    for user in self.scheduler.name_to_node[buf.name].users
                )
            else:
                buf_elems = get_buf_elems(buf)

            node_bytes += min(buf_elems, buf_accessed_elems) * get_dtype_size(
                buf.get_dtype()
            )

        return node_bytes

    def get_estimated_runtime(self) -> float:
        layout = None
        dtype = None
        if not self.node:
            assert self.snodes
            layout = self.snodes[0].node.get_layout()
            dtype = self.snodes[0].node.get_dtype()
        else:
            layout = self.node.get_layout()
            dtype = self.node.get_dtype()

        if "cuda" != layout.device.type:
            # default to no reordering based on runtime
            return 0

        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
        except Exception:
            return 0

        if isinstance(self, ExternKernelSchedulerNode):
            op = kernel_name_to_op.get(getattr(self.node, "kernel", ""), None)

            # if there is a resolved op, dry-run using fake mode and record flop count
            if op is not None:
                from torch._subclasses.fake_tensor import FakeTensorMode
                from torch.utils.flop_counter import FlopCounterMode

                with FakeTensorMode(), FlopCounterMode(
                    display=False
                ) as flop_counter_mode:
                    from .ir import ir_node_to_tensor

                    fake_inputs = [
                        ir_node_to_tensor(input) for input in self.node.inputs
                    ]
                    cls = self.node.__class__
                    cls.process_kernel(op, *fake_inputs, **self.node.kwargs)

                    # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
                    factor = 0.5
                    counted_flops = flop_counter_mode.get_total_flops()
                    return factor * counted_flops / gpu_flops

        elif isinstance(self, FusedSchedulerNode) or isinstance(
            self.node, ComputedBuffer
        ):
            return self.get_read_write_buffers_sizes() / gpu_memory_bandwidth

        # TODO(xmfan): add support for CollectiveKernel

        return 0


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def debug_str_extra(self) -> str:
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'kernel', None)}"

    def is_extern(self):
        return True

    def has_side_effects(self):
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False

        if read_dep.name not in self.scheduler.name_to_node:
            # don't allow reuse of an 'input' buffer, we don't own it
            # (would this have been fixed if I tracked mutations properly above?)
            return False

        if not isinstance(
            self.node, (torch._inductor.ir.AllReduce, torch._inductor.ir.InPlaceHint)
        ):
            # TODO make this a property of the IR
            return False

        if len(self.read_writes.writes) == 1:
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.numbytes_hint() == write_dep.numbytes_hint()

        return False


class NopKernelSchedulerNode(BaseSchedulerNode):
    pass


class SchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group_fn):
        super().__init__(scheduler, node)
        (
            self._sizes,
            self._body,
        ) = node.simplify_and_reorder()

        self.group = (node.get_device(), group_fn(self._sizes))

        if self.is_template():
            self.set_read_writes(node.normalized_read_writes())
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=True
                )
            )

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        if self.get_aliases():
            lines.append(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            lines.append(f"{name}.mutations = {pformat(self.get_mutations())}")
        if isinstance(self._body, ir.LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))
        return "\n".join(lines)

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        return bool(self.node.get_reduction_type())

    def is_template(self):
        return isinstance(self.node, ir.TemplateBuffer)

    def run(self, *index_vars):
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def mark_run(self):
        self.allocate()

    def ranges_from_index_vars(self, index_vars):
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars):
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_read_writes(self):
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes

        def fn(index):
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])

        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> Set[str]:
        buffers_store_as_atomic_add = set()
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

    def has_atomic_add(self, check_buf):
        return check_buf in self._get_atomic_add_buffers()


class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        assert node1.scheduler is node2.scheduler
        return cls(node1.scheduler, node1.get_nodes() + node2.get_nodes())

    def __init__(self, scheduler: "Scheduler", snodes: List[SchedulerNode]):
        # NB: No need to call super().__init__() because we don't need to re-use any of its logic.
        self.snodes = snodes
        self.scheduler = scheduler
        self.node = None  # type: ignore[assignment]
        self.users = None
        self.inverse_users = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group
        self.ancestors = set.union(*[x.ancestors for x in snodes])

        self.set_read_writes(
            dependencies.ReadWrites.merge_list([x.read_writes for x in snodes])
        )

        self.unmet_dependencies = {
            dep
            for dep in set.union(*[x.unmet_dependencies for x in snodes])
            if dep.name not in self.get_names()
        } - self.read_writes.writes
        self.min_order = min([x.min_order for x in self.snodes])
        self.max_order = max([x.max_order for x in self.snodes])

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_names(self) -> Set[str]:
        return set.union(*[x.get_names() for x in self.snodes])

    def debug_str_extra(self) -> str:
        lines = [
            f"{self.get_name()}.snodes[{i}] =\n{node.debug_str()}"
            for i, node in enumerate(self.snodes)
        ]
        return textwrap.indent("\n".join(lines).rstrip(), "    ")

    def set_last_usage(
        self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
    ):
        # Set self.last_usage using the global information
        # This will be used for inter-kernel optimisations
        super().set_last_usage(future_used_buffers, mutation_real_name)
        # Set self.last_usage on the snodes
        # This will be used for optimisations within the kernel
        future_used_buffers = set()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> Set[str]:
        return set.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> Set[str]:
        return set.union(*[x.used_or_aliased_buffer_names() for x in self.snodes])

    def get_nodes(self) -> List[BaseSchedulerNode]:
        return self.snodes

    def __repr__(self):
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self):
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_template(self):
        return any(x.is_template() for x in self.snodes)

    def get_device(self):
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self):
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    @cache_on_self
    def op_counts(self):
        op_counts = collections.Counter()
        for node in self.snodes:
            op_counts.update(node.op_counts())
        return op_counts

    def has_atomic_add(self, check_buf):
        return any(
            (
                isinstance(sub_schedule_node1, SchedulerNode)
                and sub_schedule_node1.has_atomic_add(check_buf)
            )
            for sub_schedule_node1 in self.get_nodes()
        )

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: Dict[str, str]):
        raise NotImplementedError

    def add_mutation_dep(self, name):
        raise NotImplementedError

    def set_users(self, users: List["NodeUser"]):
        raise NotImplementedError

    def get_aliases(self):
        raise NotImplementedError

    def get_mutations(self):
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        raise NotImplementedError

    def allocate(self):
        raise NotImplementedError

    def can_free(self):
        raise NotImplementedError


class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """Scheduler node which consists of a list of scheduler nodes that each operate on a
    distinct tensor in a list of tensors."""

    def get_consumer_subnode_for(self, producer):
        if producer.get_name() in self.read_to_node:
            return self.read_to_node[producer.get_name()]

        return None

    def get_producer_subnode_for(self, consumer):
        for rd in consumer.read_writes.reads:
            if rd.name in self.name_to_node:
                return self.name_to_node[rd.name]

        return None

    @classmethod
    def can_fuse(cls, producer, consumer):
        if producer.is_foreach() and consumer.is_foreach():
            return len(producer.snodes) == len(consumer.snodes) and all(
                producer.scheduler.can_fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            return False

        elif producer.is_foreach():
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    def fuse(cls, producer, consumer):
        assert producer.is_foreach() or consumer.is_foreach()
        prev_node_1 = None
        prev_node_2 = None
        if producer.is_foreach() and consumer.is_foreach():
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]
        elif producer.is_foreach():
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

        return cls(producer.scheduler, fused_nodes, prev_node_1, prev_node_2)

    def __init__(
        self,
        scheduler: "Scheduler",
        nodes: List[SchedulerNode],
        prev_node_1=None,
        prev_node_2=None,
    ):
        self.read_to_node = {}
        self.name_to_node = {}

        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, nodes)

            for node in nodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                for name in node.get_names():
                    self.name_to_node[name] = node
        else:
            self.scheduler = scheduler
            self.snodes = nodes

            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            self.unmet_dependencies = {
                dep
                for dep in set.union(
                    prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                )
                if dep.name not in self.get_names()
            } - self.read_writes.writes

            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])

            foreach_node = prev_node_1 if prev_node_1.is_foreach() else prev_node_2
            other_node = prev_node_2 if prev_node_1.is_foreach() else prev_node_1

            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_names():
                self.name_to_node[name] = other_node

        self.group = (nodes[0].get_device(), 0)

        self.origins = set()

    def mark_run(self):
        raise NotImplementedError

    def codegen(self):
        self.node.get_store_function()(self.node.make_loader()())

    def can_free(self):
        return NotImplementedError

    def is_foreach(self):
        return True

    def get_subkernel_nodes(self):
        """Returns a list of nodes which comprise the foreach kernel, operating on corresponding elements of our input lists.
        These nodes may be vertically fused."""
        return list(self.snodes)

    def get_nodes(self):
        """Returns all nodes contained in this kernel, unpacking fused nodes into their constituent scheduler nodes."""
        return list(itertools.chain(*[x.get_nodes() for x in self.snodes]))

    def get_first_name(self):
        return self.snodes[0].get_first_name()


def pick_loop_order(stride_lengths, sizes, priority_idx=()):
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a, b):
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        stride_len_a = [sl[a] for sl in stride_lengths]
        stride_len_b = [sl[b] for sl in stride_lengths]

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
    node: BaseSchedulerNode
    can_inplace: bool = False

    # A weak user must be scheduled after a given node, but doesn't actually
    # use the result
    is_weak: bool = False

    def get_name(self):
        return self.node.get_name()

    def merge(self, other: "NodeUser") -> "NodeUser":
        assert self.node is other.node
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


class Scheduler:
    @dynamo_timed
    def __init__(self, nodes):
        super().__init__()
        self.backends = {}
        self.fuse_cache = {}

        self.nodes = []
        self.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }

        self.nodes = [self.create_scheduler_node(n) for n in nodes]

        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        self.name_to_node = {n.get_name(): n for n in self.nodes}
        self.name_to_fused_node = None  # set in fuse_nods()

        # we handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        self.mutation_real_name = {}
        # mutation_real_name: maps back to the original name for codegen
        self.mutation_renames = {}

        self.compute_dependencies()
        self.topological_sort_schedule()
        self.dead_node_elimination()
        self.compute_ancestors()

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        V.debug.ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.create_foreach_nodes()
        self.topological_sort_schedule()
        self.fuse_nodes()
        self.compute_last_usage()
        V.debug.ir_post_fusion(self.nodes)
        V.debug.graph_diagram(self.nodes)
        self.debug_draw_graph()

        # used during codegen:
        self.current_device = None
        self.buffer_names_to_free = set()

        # fx graph node to the position it appears in the graph
        # for debug attribution
        self.origin_to_index = {}

        log.info("Number of scheduler nodes after fusion %d", len(self.nodes))

    def debug_draw_graph(self):
        """Generate an image of the graph for debugging"""
        if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
            from .debug import draw_buffers

            draw_buffers(self.nodes, print_graph=True)

    def debug_print_nodes(self, label):
        if log.isEnabledFor(logging.INFO):
            log.info("%s:", label)
            for node in self.nodes:
                node.log_details()

    def create_scheduler_node(self, node):
        assert (
            node.origins is not None
        ), "All nodes passed to scheduling must have an origin"
        if node.is_no_op():
            return NopKernelSchedulerNode(self, node)
        elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            group_fn = self.get_backend(node.get_device()).group_fn
            return SchedulerNode(self, node, group_fn)
        elif isinstance(node, ir.ExternKernel):
            return ExternKernelSchedulerNode(self, node)
        else:
            raise NotImplementedError(node)

    def create_foreach_nodes(self):
        removed_node_names = set()
        fe_nodes = []
        kept_node_names = self.name_to_fused_node.keys()

        for names in V.graph.lists.values():
            removed_node_names.update(names)

            names = [name for name in names if name in kept_node_names]
            if not names:
                # All nodes eliminated
                continue

            snodes = [self.name_to_node[name] for name in names]
            fe_node = ForeachKernelSchedulerNode(self, snodes)

            fe_nodes.append(fe_node)

            for name in names:
                self.name_to_fused_node[name] = fe_node

        self.nodes = [
            node for node in self.nodes if node.get_name() not in removed_node_names
        ] + fe_nodes

    def compute_dependencies(self):
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """
        name_to_users = collections.defaultdict(list)

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node1 in self.nodes:
            node1_name = node1.get_name()
            for node2_name in node1.get_aliases():
                if node1_name in name_to_users and node2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[node1_name]
                    list2 = name_to_users[node2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if name_to_users[key] is list1 or name_to_users[key] is list2:
                            name_to_users[key] = combined
                elif node1_name in name_to_users:
                    name_to_users[node2_name] = name_to_users[node1_name]
                else:
                    name_to_users[node1_name] = name_to_users[node2_name]

        def rename(n):
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def dep_closure(node_name):
            reachable_names = {node_name}
            node = self.name_to_node[node_name]
            write_dep = list(node.read_writes.writes)[0]
            for read_dep in node.read_writes.reads:
                if (
                    read_dep.name in self.name_to_node
                    and isinstance(read_dep, dependencies.MemoryDep)
                    and isinstance(write_dep, dependencies.MemoryDep)
                    and read_dep.index == write_dep.index
                    and read_dep.size == write_dep.size
                ):
                    reachable_names.update(dep_closure(read_dep.name))
            return reachable_names

        def add_user(used_by_name, user_node, can_inplace=False, is_weak=False):
            name_to_users[rename(used_by_name)].append(
                NodeUser(user_node, can_inplace, is_weak)
            )

        unbacked_symbol_to_origin_node = {}

        for node in self.nodes:
            # unbacked symbols don't follow ordinary buffer dependencies, so
            # we track their def/uses separately
            for s in node.node.get_unbacked_symbol_defs():
                assert s not in unbacked_symbol_to_origin_node
                unbacked_symbol_to_origin_node[s] = node

            # if a kernel takes unbacked symints, register dependencies
            for s in node.node.get_unbacked_symbol_uses():
                # NB: This is not actually a mutation dep, but we do need to
                # record this ordering dependency
                assert (
                    s in unbacked_symbol_to_origin_node
                ), f"{s} not in {unbacked_symbol_to_origin_node}"
                node.add_mutation_dep(
                    StarDep(unbacked_symbol_to_origin_node[s].get_name())
                )

            # a node will mutate either 0 or 1 buffers
            for alt_name in node.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_mutation_dep(StarDep(alt_name))
                for other_node in name_to_users[alt_name]:
                    # this node must run after all prior readers
                    other_name = rename(other_node.get_name())
                    known_dep_node_names = dep_closure(node.get_name())
                    if other_name not in known_dep_node_names:
                        # If this node already directly or indirectly depends on other_node,
                        # we don't need to insert an extra dep.
                        node.add_mutation_dep(WeakDep(other_name))
                        add_user(other_name, node, is_weak=True)

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                is_weak = isinstance(read, WeakDep)
                add_user(read.name, node, node.can_inplace(read), is_weak)

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for alt_name in node.get_mutations():
                self.mutation_renames[rename(alt_name)] = node.get_name()
                self.mutation_renames[alt_name] = node.get_name()
                self.mutation_real_name[node.get_name()] = self.mutation_real_name.get(
                    alt_name, alt_name
                )

        # make sure outputs aren't dead-code-eliminated
        for node_name in V.graph.get_output_names():
            add_user(node_name, OutputNode(StarDep(node_name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)

        inp_names = {
            name: index for index, name in enumerate(V.graph.graph_inputs.keys())
        }
        V.graph.mutated_input_idxs = [
            inp_names[name] for name in V.graph.mutated_inputs
        ]

        # copy users information onto the nodes
        for node in self.nodes:
            node.set_users(name_to_users[node.get_name()])

        # populate inverse_users
        for node in self.nodes:
            for user in node.users:
                user.node.inverse_users.append(node)

    def dead_node_elimination(self):
        """
        Remove any nodes without users
        """
        again = True  # repeat until a fixed point
        while again:
            updated_nodes = []
            for node in self.nodes:

                def can_eliminate_user(user: NodeUser):
                    return user.is_weak or user.get_name() in V.graph.removed_buffers

                can_eliminate = not node.has_side_effects() and all(
                    can_eliminate_user(u) for u in node.users
                )

                if not can_eliminate:
                    updated_nodes.append(node)
                else:
                    # dead code
                    log.debug("removed dead node: %s", node.get_name())
                    V.graph.removed_buffers.add(node.get_name())

            again = len(self.nodes) > len(updated_nodes)
            self.nodes = updated_nodes

        # Prune any WeakDeps no longer needed
        for node in self.nodes:
            node.prune_weak_deps()

    def topological_sort_schedule(self):
        """
        Ensure self.nodes is in topologically sorted order
        """
        seen = set()
        name_to_node = dict()
        result = []

        def visit(n):
            if n not in seen:
                seen.add(n)
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    visit(name_to_node[dep.name])
                result.append(n)

        for node in self.nodes:
            for name in node.get_names():
                name_to_node[name] = node
        for node in self.nodes:
            visit(node)
        self.nodes = result

    def compute_ancestors(self):
        """
        Populate each node.ancestors
        """
        # note self.nodes is topologically sorted
        name_to_ancestors = {}
        for node in self.nodes:
            ancestors = set()
            for dep in node.unmet_dependencies:
                ancestors.add(dep.name)
                ancestors |= name_to_ancestors[dep.name]
            name_to_ancestors[node.get_name()] = ancestors
            node.ancestors = ancestors

        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order

    def fuse_nodes(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.
        """
        for _ in range(10):
            old_len = len(self.nodes)
            self.fuse_nodes_once()
            if len(self.nodes) == old_len:
                break

    def fuse_nodes_once(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuses(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = set(self.nodes)
        for node1, node2 in self.get_possible_fusions():
            node1 = self.name_to_fused_node[node1.get_first_name()]
            node2 = self.name_to_fused_node[node2.get_first_name()]
            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                node3 = fuse(node1, node2)
                fused_nodes.remove(node1)
                fused_nodes.remove(node2)
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )
        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.topological_sort_schedule()
        self.prune_redundant_deps()

    def prune_redundant_deps(self):
        for node in self.nodes:
            node.prune_redundant_deps(self.name_to_fused_node)

    def get_possible_fusions(self):
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen = set()

        def check_all_pairs(nodes):
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
        for node in self.nodes:
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.aggressive_fusion:
            group_grouping = collections.defaultdict(list)
            for node in self.nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        return sorted(possible_fusions, key=self.score_fusion_key, reverse=True)

    def will_fusion_create_cycle(self, node1, node2):
        """
        Finds whether there's a path from node1 to node2 (or vice-versa)
        caused indirectly by other fusions.
        """

        def found_path(node):
            # only fused nodes can introduce new ancestors.
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                if node.get_names().issubset(combined_ancestors):
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

        visited = set()
        combined_names = node1.get_names() | node2.get_names()
        combined_ancestors = (node1.ancestors | node2.ancestors) - combined_names
        return any(found_path(self.name_to_fused_node[n]) for n in combined_ancestors)

    def can_fusion_increase_peak_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ):
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heursitic would be to use live
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

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """

        if node1 is node2:
            return False
        if (
            isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node1.is_template()
        ):
            return False
        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            return False

        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.can_fuse(node1, node2)

        if node2.get_names() & node1.ancestors:
            return False  # node1 must go before node2

        if (
            isinstance(node1, (FusedSchedulerNode, SchedulerNode))
            and isinstance(node2, SchedulerNode)
            and isinstance(node2._body, ir.LoopBody)
        ):
            # Fix issue: https://github.com/pytorch/pytorch/issues/108963
            # Check:
            #   If node2 reads a buf which is a mutation buf of node1(SchedulerNode) or among nodes in node1(FusedSchedulerNode),
            #   we will get the corresponding mutation buf and check if this mutation buf is stored by atomic_add mode.
            # If True, we will disable the fusion of node1 and node2.
            if any(
                (
                    node2_used_buf in self.mutation_renames
                    and node1.has_atomic_add(self.mutation_renames[node2_used_buf])
                )
                for node2_used_buf in node2._body.reads_name2expr.keys()
            ):
                return False

        if node2.is_template():
            return False  # only epilogues
        if node1.is_template() and (
            node2.has_aliasing_or_mutation()
            or node2.is_reduction()
            or not config.epilogue_fusion
        ):
            return False

        device = node1.get_device()
        if device != node2.get_device():
            return False  # wrong device

        no_shared_data = self.score_fusion_memory(node1, node2) == 0
        if no_shared_data and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            return False  # heuristic not needed for correctness

        if (
            not node1.is_foreach()
            and not node2.is_foreach()
            and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
        ):
            return False  # heuristic not needed for correctness

        if node1.get_names() & node2.ancestors:
            # node2 depends on node1 outputs
            if not self.can_fuse_vertical(node1, node2):
                return False
            return self.get_backend(device).can_fuse_vertical(node1, node2)
        else:  # nodes don't depend on each other, but may have common reads
            if self.can_fusion_increase_peak_memory(node1, node2):
                return False
            return self.get_backend(device).can_fuse_horizontal(node1, node2)

    def can_fuse_vertical(self, node1, node2):
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.
        """
        node1_names = node1.get_names()
        computed_deps = set()

        for rd in node2.unmet_dependencies:
            for cd in node1.read_writes.writes:
                # StarDep doesn't match MemoryDep, different indices don't match
                # However, broadcasting sometimes strips dimensions, and if that's the case
                # we still can match unmet dep
                # if there's indirect indexing, don't match it
                if (
                    rd.name == cd.name
                    and type(rd) == type(cd)
                    and not free_symbol_has(rd.index, "tmp")
                    and not free_symbol_has(cd.index, "tmp")
                    and rd.index == cd.index
                    and len(rd.size) >= len(cd.size)
                    and rd.size[: len(cd.size)] == cd.size
                ):
                    computed_deps.add(rd)

        remaining_deps = {dep.name for dep in node2.unmet_dependencies - computed_deps}
        if remaining_deps & node1_names:
            # MemoryDeps didn't match and read different locations of the same buffer.
            # Examples here include:
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            return False
        for name in remaining_deps:
            if node1_names & self.name_to_fused_node[name].ancestors:
                return False
        return True

    def score_fusion(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
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

    def score_fusion_memory(self, node1, node2):
        """
        The first term in our fusion score that estimates number of saved memory operations.
        """
        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        return sum(dep.numbytes_hint() for dep in common_memory_deps)

    def score_fusion_key(self, nodes):
        """
        Shim for list.sort(key=...)
        """
        node1, node2 = nodes
        return self.score_fusion(node1, node2)

    def compute_last_usage(self):
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

        future_used_buffers = set()
        for node_name in V.graph.get_output_names():
            future_used_buffers.add(node_name)

        for node in reversed(self.nodes):
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)

    def free_buffers(self):
        """Free any buffers that are no longer needed"""
        for name in sorted(
            self.buffer_names_to_free
            - V.graph.removed_buffers
            - V.graph.wrapper_code.freed
        ):
            if name in self.name_to_node:
                node = self.name_to_node[name]
                if node.can_free():
                    V.graph.wrapper_code.codegen_free(node.node)
            elif name in V.graph.graph_inputs:
                storage = V.graph.graph_inputs[name].data
                assert storage.is_input_buffer()
                V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

    def remove_kernel_local_buffers(self):
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """

        # V.kernel.store_buffer_names should represent the set of nodes
        # get fused
        fused_node_names = V.kernel.store_buffer_names
        names_to_remove = []
        for out_buf in V.kernel.store_buffer_names:
            users = {
                user.get_name()
                for user in self.name_to_node[out_buf].users
                if not user.is_weak
            }
            if users.issubset(fused_node_names):
                names_to_remove.append(out_buf)

        def remove_filter(n):
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
                V.graph.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name):
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        V.kernel.args.output_buffers[name] = "REMOVED"
        V.kernel.removed_buffers.add(name)

    def remove_inplace_buffer(self, name):
        log.debug("removing_inplace_buffer(%r)", name)
        inner_name = V.kernel.args.inplace_buffers[name].inner_name
        V.kernel.args.inplace_buffers[name] = inner_name.replace(
            "in_out_ptr", "REMOVED"
        )
        V.kernel.removed_buffers.add(name)

    def flush(self):
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode):
        assert isinstance(scheduler_node, ExternKernelSchedulerNode)
        # 'decide_inplace_update' stores the inplace update decisions in
        # the current kernel from where 'allocate' retrieve those decisions.
        # We have to make sure there is a non-NULL kernel handler to store
        # those inplace update decisions.
        with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
            scheduler_node.decide_inplace_update()
            scheduler_node.allocate()
        node = scheduler_node.node
        node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    def create_backend(self, device: torch.device):
        assert (
            device.type != "cuda" or device.index is not None
        ), f"{device} should have been normalized in lowering"
        V.graph.device_types.add(device.type)
        V.graph.add_device_idx(device.index)

        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        if device.type == "cuda" and not has_triton():
            device_props = torch.cuda.get_device_properties(device)
            if device_props.major < 7:
                raise RuntimeError(
                    f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
                )
            else:
                raise RuntimeError(
                    "Cannot find a working triton installation. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
                )

        return device_scheduling(self)

    def get_backend(self, device: torch.device):
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def enter_context(self, node):
        def get_order(n):
            if n not in self.origin_to_index:
                self.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return self.origin_to_index[n]

        origins = [(get_order(e), e) for n in node.get_nodes() for e in n.node.origins]
        if origins:
            _, last = max(origins)
            V.graph.wrapper_code.enter_context(last)

    @dynamo_timed
    def codegen(self):
        for node in self.nodes:
            self.enter_context(node)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if device.type == "cuda":
                        if self.current_device and self.current_device.type == "cuda":
                            V.graph.wrapper_code.codegen_device_guard_exit()
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_device_guard_enter(device.index)
                    elif self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    self.current_device = device

            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                if isinstance(node.node, ir.CUDATemplateBuffer):
                    from .codegen.cuda.cuda_scheduling import CUDAScheduling

                    CUDAScheduling(self).codegen_template(node, epilogue)
                else:
                    self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                self.codegen_extern_call(node)
            elif node.is_foreach():
                self.get_backend(device).codegen_foreach(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_nodes(node.get_nodes())
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.allocate()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_names())

        self.flush()

    def is_unaligned_buffer(self, buf_name):
        if buf_name in V.graph.graph_inputs or buf_name in V.graph.constants:
            # all graph inputs or constants are assumed to be aligned
            return False
        node = self.name_to_node[buf_name]
        layout = node.node.get_layout()
        if isinstance(layout, ir.AliasedLayout):
            return not layout.maybe_guard_aligned()
        else:
            return False


class BaseScheduling:
    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        raise NotImplementedError()

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        raise NotImplementedError()

    def group_fn(self, sizes):
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError()

    def codegen_template(
        self, template_node: BaseSchedulerNode, epilogue_nodes: List[BaseSchedulerNode]
    ):
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError()

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError()

    def codegen_sync(self):
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
        raise NotImplementedError()

    def flush(self):
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        raise NotImplementedError()
