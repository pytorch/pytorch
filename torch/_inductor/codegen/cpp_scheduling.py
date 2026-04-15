# mypy: allow-untyped-defs
import itertools
import math
import sys
from collections.abc import Sequence
from enum import Enum
from typing import Any, cast
from typing_extensions import override

import sympy

import torch
from torch._inductor import dependencies
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ..._dynamo.utils import counters
from .. import config, ir, metrics
from ..debug import set_kernel_post_grad_provenance_tracing
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    ExternKernelSchedulerNode,
    ForeachKernelSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from ..utils import get_fused_kernel_name, is_multi_outputs_template, Placeholder
from ..virtualized import V
from .common import BackendFeature, IndentedBuffer
from .cpp import (
    CppKernelProxy,
    KernelGroup,
    OuterLoopFusedSchedulerNode,
    stride_at_vec_range,
)
from .cpp_utils import LocalBufferContext, template_fusion_with_epilogues_supported


_IS_WINDOWS = sys.platform == "win32"


class ReasonFusedNodes(Enum):
    SAME_VARS_REDUCE = "same_vars_reduce"
    COMPATIBLE_REDUCTION = "compatible_reduction"
    COMPATIBLE_RANGES_NO_REDUCTION = "compatible_ranges_no_reduction"


class CppScheduling(BaseScheduling):
    """Schedule and fuse C++ Inductor kernels for CPU code generation."""

    # Subclass CppKernelProxy to customize codegen without copying codegen_node().
    # Use kernel_proxy_cls to inject custom proxies in CppScheduling subclasses.
    # Avoid duplicating codegen_node() just to swap in a custom kernel proxy class.
    kernel_proxy_cls: type[CppKernelProxy] = CppKernelProxy
    # ctypes limits the number of args to 1024, refer to:
    # https://github.com/python/cpython/commit/a285af7e626d1b81cf09f8b2bf7656f100bc1237
    # We set a conservative threshold here.
    MAX_FUSED_KERNEL_ARGS_NUM = 500
    backend_features = OrderedSet(
        [
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
        ]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def __init__(self, scheduler):
        super().__init__(scheduler)
        if scheduler:
            self.reset_kernel_group()
        self._ready_to_flush = False

    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def reset_kernel_group(self):
        self.kernel_group = KernelGroup()

    @override
    def fuse(self, node1, node2):
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        elif node1.is_template():
            assert not node2.is_template()
            return FusedSchedulerNode.fuse(node1, node2)
        else:
            if (
                self._why_fuse_nodes(node1, node2)
                == ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
            ):
                assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
                assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))

                _, (vars1, reduce1) = node1.group
                _, (vars2, reduce2) = node2.group
                assert reduce1 == () and reduce2 == (), (reduce1, reduce2)

                def get_indexing_ranges_exprs(node):
                    if isinstance(node, FusedSchedulerNode):
                        assert len(node.snodes) > 0, node.snodes
                        var_ranges = None
                        indexing_exprs = OrderedSet[Any]()
                        for snode in node.snodes:
                            v, exprs = get_indexing_ranges_exprs(snode)
                            if var_ranges is None:
                                var_ranges = v
                            assert var_ranges == v, (var_ranges, v, node.snodes)
                            indexing_exprs.update(exprs)
                        return var_ranges, list(indexing_exprs)
                    else:
                        assert isinstance(node, SchedulerNode)
                        comp_buffer = node.node
                        assert isinstance(comp_buffer, ir.ComputedBuffer)
                        _, body, _ = comp_buffer.get_default_sizes_body()
                        return body.var_ranges, list(body.indexing_exprs.values())

                node_to_recomp = node1 if len(vars1) < len(vars2) else node2
                assert isinstance(node_to_recomp, SchedulerNode)

                ref_node = node2 if len(vars1) < len(vars2) else node1

                ref_indexing_constraints = get_indexing_ranges_exprs(ref_node)

                node_to_recomp.recompute_size_and_body(
                    extra_indexing_constraints=ref_indexing_constraints
                )

                _, (vars1, _) = node1.group
                _, (vars2, _) = node2.group

                if vars1 == vars2:
                    return FusedSchedulerNode.fuse(node1, node2)

                # recompute ref_node if its ranges are also changed
                node_to_recomp_indexing_constraints = get_indexing_ranges_exprs(
                    node_to_recomp
                )
                if isinstance(ref_node, SchedulerNode):
                    ref_node.recompute_size_and_body(
                        extra_indexing_constraints=node_to_recomp_indexing_constraints
                    )
                else:
                    assert isinstance(ref_node, FusedSchedulerNode)
                    for snode in ref_node.snodes:
                        assert isinstance(snode, SchedulerNode)
                        snode.recompute_size_and_body(
                            extra_indexing_constraints=node_to_recomp_indexing_constraints
                        )
                    ref_node = FusedSchedulerNode(ref_node.scheduler, ref_node.snodes)

                _, (vars1, _) = node1.group
                _, (vars2, _) = node2.group
                assert vars1 == vars2, (vars1, vars2)
                return FusedSchedulerNode.fuse(node1, node2)
            elif self.can_fuse_vertical_outer_loop(node1, node2):
                return OuterLoopFusedSchedulerNode.fuse(
                    node1, node2, self._get_outer_loop_fusion_depth(node1, node2)
                )
            else:
                return FusedSchedulerNode.fuse(node1, node2)

    def _why_fuse_nodes(self, node1, node2) -> ReasonFusedNodes | None:
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        if vars1 == vars2 and reduce1 == reduce2:
            return ReasonFusedNodes.SAME_VARS_REDUCE
        if reduce1 == () and vars1 == vars2 + reduce2:
            return ReasonFusedNodes.COMPATIBLE_REDUCTION
        if self._can_fuse_nodes_with_compatible_ranges(node1, node2):
            return ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return None

    def _can_fuse_nodes_with_compatible_ranges(self, node1, node2):
        # Here we try to fuse SchedulerNode/FusedSchedulerNode with compatible ranges
        # e.g. (s0, s1, s2) and (s0 * s1 * s2)
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        c1 = reduce1 == () and reduce2 == ()
        c2 = math.prod(vars1) == math.prod(vars2)
        c3 = len(vars1) == 1 or len(vars2) == 1
        if not (c1 and c2 and c3):
            return False

        node_to_recomp = node1 if len(vars1) < len(vars2) else node2
        ref_node = node2 if len(vars1) < len(vars2) else node1

        # We can not recompute sizes and body for nodes other than SchedulerNode
        # TODO: we can extend fusion support with compatible ranges for FusedSchedulerNode
        if isinstance(node_to_recomp, FusedSchedulerNode):
            return False

        # It may happen that node1 and node2 compatible number of elements
        # but different original ranges, for example:
        # {d0: s0, d1: s1, d2: s2} vs {d0: s0*s1*s2}
        # See https://github.com/pytorch/pytorch/pull/120077/files#r1500427848 for more details
        # TODO: we can fix if it allows us to CSE at least one of the variables

        assert isinstance(node_to_recomp, SchedulerNode)
        if isinstance(node_to_recomp.node, ir.TemplateBuffer):
            return False
        assert isinstance(node_to_recomp.node, ir.ComputedBuffer)
        # node.data.get_size() is a cheaper version of node.get_read_writes().var_ranges
        # but without variable name
        ranges2 = node_to_recomp.node.data.get_size()
        ranges1 = None
        if isinstance(ref_node, FusedSchedulerNode):
            ranges_set = OrderedSet[tuple[Any, ...]]()
            for snode in ref_node.snodes:
                if isinstance(snode.node, ir.TemplateBuffer):
                    break
                assert isinstance(snode.node, ir.ComputedBuffer)
                ranges_set.add(tuple(snode.node.data.get_size()))

            if len(ranges_set) != 1:
                return False

            ranges1 = list(next(iter(ranges_set)))
        else:
            assert isinstance(ref_node, SchedulerNode)
            assert isinstance(ref_node.node, ir.ComputedBuffer)
            ranges1 = ref_node.node.data.get_size()  # type: ignore[assignment]

        if ranges1 != ranges2:
            return False

        return True

    def _can_fuse_horizontal_impl(self, node1, node2):
        assert isinstance(
            node1, (FusedSchedulerNode, SchedulerNode, ExternKernelSchedulerNode)
        )
        assert isinstance(node2, (FusedSchedulerNode, SchedulerNode))
        if any(
            isinstance(node, (OuterLoopFusedSchedulerNode, ExternKernelSchedulerNode))
            for node in (node1, node2)
        ):
            return False
        return self._why_fuse_nodes(node1, node2) is not None

    def can_fuse_horizontal(self, node1, node2):
        if node1.is_template() or node2.is_template():
            return False
        if (
            len(node1.get_nodes()) + len(node2.get_nodes())
            > config.cpp.max_horizontal_fusion_size
        ):
            return False

        return self._can_fuse_horizontal_impl(node1, node2)

    def can_fuse_multi_outputs_template(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if template_buf := node1.get_template_node():
            return (
                isinstance(template_buf.layout, ir.MultiOutputLayout)
                and isinstance(node2.node, ir.MultiOutput)
                and len(node2.node.inputs) == 1
                and node2.node.inputs[0].get_name() == template_buf.name  # type: ignore[union-attr]
            )
        return False

    def _get_outer_loop_fusion_depth(self, node1, node2):
        DISABLE_OUTER_LOOP_FUSION = 0
        if not all(
            type(node)
            in (OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode)
            for node in (node1, node2)
        ):
            return DISABLE_OUTER_LOOP_FUSION

        _node1 = (
            node1.get_outer_nodes()[-1]
            if isinstance(node1, OuterLoopFusedSchedulerNode)
            else node1
        )
        assert isinstance(_node1, (FusedSchedulerNode, SchedulerNode))
        _node2 = (
            node2.get_outer_nodes()[0]
            if isinstance(node2, OuterLoopFusedSchedulerNode)
            else node2
        )
        assert isinstance(_node2, (FusedSchedulerNode, SchedulerNode))

        _, (vars1, reduce1) = _node1.group
        _, (vars2, reduce2) = _node2.group
        if vars1 == () and vars2 == () and reduce1 != () and reduce2 != ():
            # Reduction only
            return DISABLE_OUTER_LOOP_FUSION
        if all(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return (
                node1.outer_loop_fusion_depth
                if node1.outer_loop_fusion_depth == node2.outer_loop_fusion_depth
                else DISABLE_OUTER_LOOP_FUSION
            )
        outer_loop_fusion_depth = min(len(vars1), len(vars2))
        if (
            outer_loop_fusion_depth >= 1
            and vars1[:outer_loop_fusion_depth] == vars2[:outer_loop_fusion_depth]
        ):
            if any(
                type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)
            ):
                _compare_node = (
                    node1 if type(node1) is OuterLoopFusedSchedulerNode else node2
                )
                if _compare_node.outer_loop_fusion_depth == outer_loop_fusion_depth:
                    # Same outer loop fusion depth as prev nodes in OuterLoopFusedSchedulerNode
                    return outer_loop_fusion_depth
                else:
                    return DISABLE_OUTER_LOOP_FUSION
            else:
                # First 2 nodes to generate OuterLoopFusedSchedulerNode
                return outer_loop_fusion_depth
        return DISABLE_OUTER_LOOP_FUSION

    def can_fuse_vertical_outer_loop(self, node1, node2):
        return (
            not node1.is_template()
            and not node2.is_template()
            and node1.get_operation_names() & node2.ancestors
            and not (
                self._can_fuse_horizontal_impl(node1, node2)
                and not node1.is_reduction()
            )
            and self._get_outer_loop_fusion_depth(node1, node2) >= 1
        )

    def get_fusion_pair_priority(self, node1, node2):
        if self.can_fuse_vertical_outer_loop(node1, node2):
            # Outer loop fusion with lower priority
            return 1
        else:
            return 0

    def can_fuse_vertical(self, node1, node2):
        if node2.is_template():
            # TODO(jgong5): support pre-op fusion with template
            return False
        if node1.is_template():
            template_fusion_supported, _ = template_fusion_with_epilogues_supported(
                node1, [node2]
            )
            return not node2.is_reduction() and template_fusion_supported
        return (
            self._can_fuse_horizontal_impl(node1, node2) and not node1.is_reduction()
        ) or self.can_fuse_vertical_outer_loop(node1, node2)

    def try_loop_split(self, nodes: list[SchedulerNode]):
        """
        Apply loop split optimization.
        When one of the indexing_exprs contains a division, we eliminate the division by splitting the loop
        to avoid non-contiguous loads, subject to the following conditions:
            1. No reduction and no mudular index for all nodes.
            2. The indexing_exprs of all nodes contain only one (or more, but all the same) division,
               where the divisor is an integer and not too small (the divisor > 8), the dividend is
               one of the iter_vars, and this var, i.e. the dimension that needs to be split, is
               contiguous in all other indexing_exprs.

        For example, if the node's var_ranges: {z0: 2, z1: 9216, z2: 960} and indexing_exprs:
        {'index0': 8847360*z0 + 960*z1 + z2, 'index1': 32*z0 + (z2//30), 'index2': z2},
        we will split z2 -> 30*z2 + z3, then the node's var_ranges will be changed to
        {z0: 2, z1: 9216, z2: 32, z3: 30} and indexing_exprs will be changed to
        {'index0': 8847360*z0 + 960*z1 + 30*z2 + z3, 'index1': 32*z0 + z2, 'index2': 30*z2 + z3}.
        """

        # No reduction and no mudular
        if any(
            len(node.group[1][1]) != 0
            or any(
                expr.has(ModularIndexing) for expr in node._body.indexing_exprs.values()
            )
            for node in nodes
        ):
            return nodes

        split_var = None
        split_number = None
        num_div = 0
        div_expr_ = None
        match_div = False
        matched_node = None
        matched_index_size = None

        # Collect node info for later compatibility check
        node_bodies: list[tuple[Any, Any]] = []

        for node in nodes:
            assert isinstance(node.node, ir.ComputedBuffer)
            sizes_body = node.node.get_default_sizes_body()
            node_bodies.append((node, sizes_body))
            (index_size, _), original_body, _ = sizes_body
            for name, expr in original_body.indexing_exprs.items():
                if not isinstance(expr, sympy.Expr):
                    continue
                for div_expr in expr.find(FloorDiv):
                    if (
                        any(div_expr.has(var) for var in original_body.iter_vars)
                        and div_expr != div_expr_
                    ):
                        div_expr_ = div_expr
                        num_div += 1
                    if num_div > 1:
                        return nodes
                    if (
                        isinstance(div_expr.args[1], sympy.core.numbers.Integer)
                        and div_expr.args[0] in original_body.iter_vars
                        and name is not None
                        and all(
                            stride_at_vec_range(expr_, div_expr.args[0]) in (0, 1)
                            for name_, expr_ in original_body.indexing_exprs.items()
                            if name_ != name
                        )
                        and div_expr.args[1] > 8
                    ):
                        split_var = div_expr.args[0]
                        split_number = div_expr.args[1]
                        match_div = True
                        matched_node = node
                        matched_index_size = index_size

        # Only one node contains a division, and the split dimension is contiguous in all other indexing_exprs.
        if not match_div:
            return nodes

        # Check if all nodes have split_var in their iter_vars and have compatible sizes
        # (same number of index dimensions). If not, bail out to avoid incompatible
        # var_ranges after loop split which would cause assertion failures in
        # simplify_and_reorder or codegen_functions.
        assert matched_index_size is not None
        matched_num_dims = len(matched_index_size)

        for node, ((index_size, _), original_body, _) in node_bodies:
            if split_var not in original_body.iter_vars:
                return nodes
            if len(index_size) != matched_num_dims:
                return nodes

        extra_indexing_constraints = None

        def loop_split(sizes, body, vars):
            index_size, reduce_size = sizes
            index_vars, reduce_vars = vars
            split_idx = index_vars.index(split_var)
            new_index_size = index_size.copy()
            new_index_size[split_idx] = index_size[split_idx] // split_number
            new_index_size.insert(split_idx + 1, split_number)
            (new_index_vars, _), var_ranges = dependencies.index_vars_no_squeeze(
                new_index_size, reduce_size, prefix="y"
            )
            iter_vars = new_index_vars.copy()
            divisor_var = iter_vars.pop(split_idx + 1)
            iter_vars[split_idx] = split_number * iter_vars[split_idx] + divisor_var
            body = ir.LoopBody(
                body, [iter_vars, reduce_vars], var_ranges, new_index_vars, reduce_vars
            )
            nonlocal extra_indexing_constraints
            if not extra_indexing_constraints:
                extra_indexing_constraints = (
                    body.var_ranges,
                    list(body.indexing_exprs.values()),
                )
            return (
                (new_index_size, reduce_size),
                body,
                (new_index_vars, reduce_vars),
            )

        # Here decide the final loop order
        for node in nodes:
            if node == matched_node:
                node.recompute_size_and_body(recompute_sizes_body_func=loop_split)
        for node in nodes:
            if node != matched_node:
                node.recompute_size_and_body(
                    extra_indexing_constraints=extra_indexing_constraints,
                    recompute_sizes_body_func=loop_split,
                )

        return nodes

    def codegen_outer_loop_node(
        self,
        node: OuterLoopFusedSchedulerNode,
    ):
        """
        Generate the code for the outer loop fused scheduler node.
        1. Codegen with fused outer loop: depends on the analysis of
            the outer loop fused scheduler node, with or without the local buffer.
        2. If failed, fallback to standard codegen.
        """
        kernel_group = self.kernel_group
        generated_cpp_vec_kernel_count = metrics.generated_cpp_vec_kernel_count
        cpp_kernel_proxy_list: list[self.kernel_proxy_cls] = []  # type: ignore[name-defined]
        nodes_list: list[list[SchedulerNode]] = []
        assert isinstance(node, OuterLoopFusedSchedulerNode)

        def try_outer_loop_fusion_with_local_buf(node: OuterLoopFusedSchedulerNode):
            """
            Codegen code with fused outer loop and local Buffer.
            """
            assert isinstance(node, OuterLoopFusedSchedulerNode)
            cpp_kernel_proxy_list.clear()
            nodes_list.clear()

            def get_call_ranges(node: BaseSchedulerNode):
                assert isinstance(node, (SchedulerNode, FusedSchedulerNode))
                nodes: list[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
                _, (group, reduction_group) = max(
                    nodes, key=lambda x: int(x.is_reduction())
                ).group
                call_ranges = tuple(group) + tuple(reduction_group)
                return call_ranges

            local_buffers: list[ir.Buffer] = []
            # Map local buffer name to a list of global buffers
            local_to_global_buffers: dict[str, list[ir.Buffer]] = {}
            if all(
                len(get_call_ranges(_node)) == node.outer_loop_fusion_depth + 1
                for _node in node.get_outer_nodes()
            ):
                # Ref to the typical case of local buffer in
                # https://github.com/pytorch/pytorch/blob/1115a25c36340554442f28f9570abd42f0aface2/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L159 # noqa: B950
                # where the buffer is with size of last dim and contiguous.
                # Only support this typical case at first.
                visited_scheduler_nodes: OrderedSet[str] = OrderedSet()
                for scheduler_node in node.get_nodes():
                    # all users inside same OuterLoopFusedSchedulerNode
                    assert isinstance(scheduler_node, SchedulerNode)
                    visited_scheduler_nodes.add(scheduler_node.get_name())
                    if (
                        scheduler_node.is_reduction()
                        or len(scheduler_node.get_outputs()) != 1
                    ):
                        continue

                    scheduler_buffer = scheduler_node.get_outputs()[0]
                    if all(
                        user.node in node.get_nodes() for user in scheduler_buffer.users
                    ):
                        global_buffer = scheduler_buffer.node
                        assert isinstance(global_buffer, ir.ComputedBuffer)
                        global_buffer_layout = global_buffer.get_layout()
                        size_offset = node.outer_loop_fusion_depth - len(
                            get_call_ranges(scheduler_node)
                        )

                        def is_all_write_read_contiguous():
                            contiguous_index_expr = 0
                            stride = 1
                            for var, range in reversed(
                                # pyrefly: ignore [missing-attribute]
                                scheduler_node._body.var_ranges.items()
                            ):
                                contiguous_index_expr += stride * var
                                stride *= range
                            # pyrefly: ignore [missing-attribute]
                            write_index_expr = scheduler_node._body.get_write_expr(
                                scheduler_buffer.get_name()
                            )

                            def is_contiguous_index(x):
                                return x == contiguous_index_expr

                            return is_contiguous_index(write_index_expr) and all(
                                isinstance(user.node, SchedulerNode)
                                and is_contiguous_index(
                                    user.node._body.get_read_expr(
                                        scheduler_buffer.get_name()
                                    ),
                                )
                                for user in scheduler_buffer.users
                            )

                        if not (
                            global_buffer_layout.is_contiguous()
                            and is_all_write_read_contiguous()
                        ):
                            continue
                        # Local Buffer is a view of global buffer
                        local_buffer_stride: list[int] = []
                        stride = global_buffer_layout.stride[-1]
                        local_buffer_size = get_call_ranges(scheduler_node)[
                            size_offset:
                        ]
                        for sz in reversed(local_buffer_size):
                            local_buffer_stride.insert(0, stride)
                            stride *= sz
                        local_buffer_layout = ir.FixedLayout(
                            global_buffer_layout.device,
                            global_buffer_layout.dtype,
                            local_buffer_size,
                            local_buffer_stride,
                        )

                        def try_share_local_buffer(local_buffer_layout, local_buffers):
                            for local_buf in local_buffers:
                                if local_buffer_layout == local_buf.layout and all(
                                    all(
                                        user.node.get_name() in visited_scheduler_nodes
                                        for user in V.graph.scheduler.name_to_buf[
                                            global_buffer.name
                                        ].users
                                    )
                                    for global_buffer in local_to_global_buffers[
                                        local_buf.name
                                    ]
                                    if global_buffer.name is not None
                                ):
                                    return local_buf
                            return None

                        local_buf_prefix = "local_buffer_data"
                        # Share existing local buffer
                        local_buffer_used = try_share_local_buffer(
                            local_buffer_layout, local_buffers
                        )
                        if not local_buffer_used:
                            # Create new local buffer
                            local_buffer_used = ir.Buffer(
                                name=f"{local_buf_prefix}_{len(local_buffers)}",
                                layout=local_buffer_layout,
                            )
                            local_buffers.append(local_buffer_used)
                            local_to_global_buffers[local_buffer_used.name] = []  # type: ignore[index]

                        local_to_global_buffers[local_buffer_used.name].append(
                            global_buffer,
                        )

            with LocalBufferContext(kernel_group.args) as scope:
                if len(local_buffers) > 0:
                    for local_buffer in local_buffers:
                        assert local_buffer.name is not None
                        scope.add_local_buffer(
                            local_buffer, local_to_global_buffers[local_buffer.name]
                        )
                for _node in node.get_outer_nodes():
                    assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                    cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
                    cpp_kernel_proxy.codegen_nodes(_node.get_nodes())  # type: ignore[arg-type]
                    cpp_kernel_proxy_list.append(cpp_kernel_proxy)
                    nodes_list.append(_node.get_nodes())  # type: ignore[arg-type]

                if not node.check_outer_fusion_loop_level_attr(
                    cpp_kernel_proxy_list, node.outer_loop_fusion_depth
                ):
                    for removed_buffer in scope.removed_buffers:
                        # Restore the removed buffers by this context before
                        # fallback to codegen without using Local Buffer
                        V.graph.removed_buffers.remove(removed_buffer)
                    return False
                metrics.cpp_outer_loop_fused_inner_counts.append(
                    metrics.CppOuterLoopFusedCount(
                        len(cpp_kernel_proxy_list),
                        local_buffer_number=len(scope.local_buffers),
                    )
                )
                outer_fusion_cpp_kernel_proxy = node.merge_outer_fusion_kernels(
                    cpp_kernel_proxy_list,
                )
                kernel_group.finalize_kernel(
                    outer_fusion_cpp_kernel_proxy,
                    [*itertools.chain.from_iterable(nodes_list)],
                )

            return True

        if not try_outer_loop_fusion_with_local_buf(node):
            # Reset generated_cpp_vec_kernel_count to codegen again
            metrics.generated_cpp_vec_kernel_count = generated_cpp_vec_kernel_count
            cpp_kernel_proxy_list.clear()
            nodes_list.clear()
            # Similar as comment in
            # https://github.com/pytorch/pytorch/blob/469383755fe416eb1c41fa724762ad3eaecdff07/torch/_inductor/codegen/cpp.py#L3269-L3272
            # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
            with torch._inductor.config.patch(inplace_buffers=False):
                for _node in node.get_outer_nodes():
                    assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                    _nodes: list[SchedulerNode] = _node.get_nodes()  # type: ignore[assignment]
                    cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
                    cpp_kernel_proxy.codegen_nodes(_nodes)
                    kernel_group.finalize_kernel(cpp_kernel_proxy, _nodes)

    def codegen_node(
        self,
        node: OuterLoopFusedSchedulerNode | FusedSchedulerNode | SchedulerNode,
    ):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        if isinstance(node, OuterLoopFusedSchedulerNode):
            self.codegen_outer_loop_node(node)
        else:
            nodes: list[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
            nodes = self.try_loop_split(nodes)
            cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
            cpp_kernel_proxy.codegen_nodes(nodes)
            kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)

        args_num = self._get_scheduled_num_args()
        if args_num > CppScheduling.MAX_FUSED_KERNEL_ARGS_NUM:
            self._set_flush_status(True)

    def is_cpp_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ir.CppTemplateBuffer
        )

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a CPP template, possibly with fused epilogues
        """
        assert not prologue_nodes

        # remove MultiOutput from epilogue_nodes
        epilogue_nodes = [
            epilogue_node
            for epilogue_node in epilogue_nodes
            if isinstance(epilogue_node, (SchedulerNode, FusedSchedulerNode))
        ]
        # The counter cpp_templated_kernel_counter is used for verifying if a
        # a templated kernel was successfully compiled in a UT
        counters["inductor"]["cpp_templated_kernel_counter"] += 1
        counters["inductor"]["cpp_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cpp_template(template_node), (
            "Template node passed to CppScheduler.codegen_template must be a SchedulerNode that wraps a CppTemplateBuffer"
        )
        template_node = cast(SchedulerNode, template_node)
        _, (_, rnumel) = template_node.group
        assert rnumel == ()
        ctb: ir.CppTemplateBuffer = cast(ir.CppTemplateBuffer, template_node.node)
        epilogue_ir_nodes: list[ir.Operation | None] = [n.node for n in epilogue_nodes]
        assert all(isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes), (
            "Epilogue nodes must all be instances of ir.ComputedBuffer"
        )

        def template_buffer_has_other_users(
            template_buffer, outputs_by_name, epilogue_nodes
        ):
            if not epilogue_nodes:
                return False

            assert template_buffer.get_name() in outputs_by_name
            users = outputs_by_name[template_buffer.get_name()].users
            return not all(
                isinstance(user.node, BaseSchedulerNode)
                and user.node.node in epilogue_nodes
                for user in users
            )

        flag_template_buffer_has_other_users = template_buffer_has_other_users(
            ctb, template_node.outputs_by_name, epilogue_ir_nodes
        )
        kernel, render = ctb.make_kernel_render(  # type: ignore[misc]
            ctb,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_ir_nodes,
        )
        with kernel:
            if not is_multi_outputs_template(template_node.node):
                template_node.mark_run()  # type: ignore[attr-defined]
            for node in epilogue_nodes:
                node.mark_run()  # type: ignore[attr-defined]
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule, kernel.args)

        if is_multi_outputs_template(template_node.node):
            # For multi outputs template, allocate buffers for each output after the epilogue
            # codegen to which determines if the buffer has been removed.
            assert len(template_node.outputs) == 1, (
                "Multi outputs template should be with 1 output template buffer of MultiOutputLayout"
            )
            for user in template_node.outputs[0].users:
                assert isinstance(user.node, ExternKernelSchedulerNode), (
                    "Multi outputs template should be with ExternKernelSchedulerNode"
                )
                assert isinstance(user.node.node, ir.MultiOutput), (
                    "Multi outputs template has multi users with MultiOutput"
                )
                user.node.mark_run()

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    def _get_scheduled_num_args(self):
        return self.kernel_group.get_num_args()

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def define_kernel(self, src_code, nodes, kernel_args=None):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(nodes, config.cpp.descriptive_names)
                if config.cpp.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cpp", fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_decl_name)
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "//")

            # Get the lines in the source code representing the function definition,
            # excluding the first line including cpp_prefix.h.
            first_char = src_code.rfind('extern "C"')
            last_char = src_code.find(")", first_char)
            if _IS_WINDOWS:
                # get_export_declaration introduced one more ')' in Windows
                last_char = src_code.find(")", last_char + 1)
            kernel_definition = f"{src_code[first_char : last_char + 1]};\n"

            compile_wrapper = IndentedBuffer()
            args = self.kernel_group.args if kernel_args is None else kernel_args
            _, _, arg_types = args.cpp_argdefs()
            if not V.graph.cpp_wrapper:
                compile_wrapper.writeline(
                    f"async_compile.cpp_pybinding({arg_types!r}, r'''"
                )
            compile_wrapper.splice(src_code, strip=True)
            if not V.graph.cpp_wrapper:
                compile_wrapper.writeline("''')")
            wrapper.define_kernel(
                kernel_name,
                compile_wrapper.getvalue(),
                gpu=False,
                cpp_definition=kernel_definition,
            )
        return kernel_name

    def flush(self):
        src_code = self.kernel_group.codegen_group()
        if src_code:
            kernel_name = self.define_kernel(
                src_code, self.kernel_group.scheduled_nodes
            )
            self.codegen_comment(self.kernel_group.scheduled_nodes, kernel_name)
            if config.cpp.enable_kernel_profile:
                V.graph.wrapper_code.write_kernel_context_guard_begin()
                V.graph.wrapper_code.write_kernel_context_guard(
                    kernel_name,
                    self.kernel_group.scheduled_nodes,  # type: ignore[arg-type]
                )
            self.kernel_group.call_kernel(V.graph.wrapper_code, kernel_name)
            if config.cpp.enable_kernel_profile:
                V.graph.wrapper_code.write_kernel_context_guard_end()

        self.reset_kernel_group()
        self._set_flush_status(False)

    def codegen_comment(self, node_schedule, kernel_name=None):
        # below add provenance tracing info for cpu CppKernel types
        wrapper = V.graph.wrapper_code
        debug_handle = set_kernel_post_grad_provenance_tracing(
            node_schedule,  # type: ignore[arg-type]
            # pyrefly: ignore [bad-argument-type]
            kernel_name,
        )
        wrapper.write_provenance_debug_handle(kernel_name, debug_handle)
