import io
import itertools
import logging
import textwrap
import tokenize
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, cast, TYPE_CHECKING

import sympy
from sympy import Integer, Symbol

import torch


if TYPE_CHECKING:
    import triton

from torch.utils._ordered_set import OrderedSet

from .. import config, metrics
from ..runtime.hints import DeviceProperties, TritonMeta
from ..runtime.runtime_utils import next_power_of_2
from ..runtime.triton_heuristics import (
    RoundRobinComboKernelGrid,
    SequentialComboKernelGrid,
    SequentialFlattenComboKernelGrid,
)
from ..scheduler import BaseSchedulerNode
from ..stream_utils import get_raw_stream_name
from ..utils import (
    clear_on_fresh_cache,
    DeferredLineBase,
    Placeholder,
    triton_type,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    IndentedBuffer,
    InplacedBuffer,
    Kernel,
    PythonPrinter,
    RemovedArg,
    SizeArg,
    TensorArg,
    WorkspaceArg,
)
from .simd import NodeInfo, prefix_is_reduction, SIMDScheduling
from .simd_kernel_features import SIMDKernelFeatures
from .triton import TritonKernel
from .triton_utils import (
    config_of,
    equal_1_arg_indices,
    is_unaligned_buffer,
    signature_to_meta,
)


# Default block sizes used when combo kernel autotuning is disabled.
DEFAULT_COMBO_BLOCK_SIZE_1D = 1024
DEFAULT_COMBO_BLOCK_SIZE_2D = 32


log = logging.getLogger(__name__)
pexpr = PythonPrinter().doprint
LARGE_NUMELS = 51_200_000
BLOCK_UTILIZATION = 0.8


def _size_hint(expr: Any) -> int:
    return V.graph.sizevars.optimization_hint(expr, fallback=1)


def _node_partition_log_context(
    node: BaseSchedulerNode, node_info_map: dict[BaseSchedulerNode, NodeInfo]
) -> tuple[Any, ...]:
    node_info = node_info_map[node]
    tiling_hints = tuple(
        (str(dim), _size_hint(numel))
        for dim, numel in sorted(
            node_info.tiling.items(), key=lambda item: str(item[0])
        )
    )
    return (
        bool(node_info.features.is_reduction()),
        node_info.is_persistent_reduction,
        tiling_hints,
        _size_hint(node_info.numel),
        _size_hint(node_info.rnumel),
    )


def _partition_separation_log_context(
    separated_nodes: list[BaseSchedulerNode],
    companion_nodes: list[BaseSchedulerNode],
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (
        tuple(_node_partition_log_context(n, node_info_map) for n in separated_nodes),
        tuple(_node_partition_log_context(n, node_info_map) for n in companion_nodes),
    )


def _log_partition_separation(
    log_message: str,
    separated_nodes: list[BaseSchedulerNode],
    companion_nodes: list[BaseSchedulerNode],
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> None:
    if log.isEnabledFor(logging.DEBUG):
        _log_partition_separation_once(
            log_message,
            len(separated_nodes),
            _partition_separation_log_context(
                separated_nodes, companion_nodes, node_info_map
            ),
        )


# This diagnostic is otherwise repeated once per equivalent recompile.
@clear_on_fresh_cache
@cache
def _log_partition_separation_once(
    log_message: str,
    num_nodes: int,
    partition_context: tuple[tuple[Any, ...], tuple[Any, ...]],
) -> None:
    log.debug(
        log_message,
        num_nodes,
    )


def _default_custom_combo_kernel_horizontal_partition(
    nodes: list[BaseSchedulerNode],
    triton_scheduling: SIMDScheduling,
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> list[list[BaseSchedulerNode]]:
    """Horizontally partition the given list of nodes into a list of list of nodes where each sublist
    represents a partition. Nodes in different partitions are implemented in different combo kernels.
    Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    Input arguments:
        nodes: a list of fused scheduler nodes to partition.
        triton_scheduling: TritonScheduling instance.
        node_info_map: a map from node to NodeInfo NamedTuple
    Output:
        a list of list of nodes with each sublist representing a partition.

    The default algorithm is to partition nodes based on the following rules:
        1) nodes with the same number of block dimensions are grouped together.
        2) large pointwise nodes (numels greater than LARGE_NUMELS) are separated from other nodes.
        3) large reduce nodes are separated from other nodes.
    """

    if len(nodes) < 1:
        raise AssertionError(f"expected at least 1 node, got {len(nodes)}")

    # first partition nodes based on number of block dimensions
    tilings = [node_info_map[n].tiling for n in nodes]

    max_dims = max(len(t) for t in tilings)
    nodes_per_ndim: list[list[BaseSchedulerNode]] = []
    for i in range(2, max_dims + 1):
        group_per_dim = [n for n, t in zip(nodes, tilings) if len(t) == i]
        reduction = [
            n for n in group_per_dim if node_info_map[n].features.is_reduction()
        ]
        not_reduction = [n for n in group_per_dim if n not in reduction]
        # rnumel > 2048 usually has long execution time
        # BaseSchedulerNode.group[-1][-1] is rnumel for reduction nodes
        # Scheduling heuristic: separate long reductions (rnumel > 2048).
        # Uses optimization_hint with fallback=1 so unbacked defaults to short reduction.
        long_reduction = [
            n
            for n in reduction
            if V.graph.sizevars.optimization_hint(n.group[-1][-1], fallback=1) > 2048  # type: ignore[arg-type]
        ]
        short_reduction = [n for n in reduction if n not in long_reduction]
        very_large_reduction = [
            n
            for n in long_reduction
            if (
                V.graph.sizevars.optimization_hint(node_info_map[n].numel, fallback=1)
                * V.graph.sizevars.optimization_hint(
                    node_info_map[n].rnumel, fallback=1
                )
            )
            > LARGE_NUMELS
        ]
        long_reduction = [n for n in long_reduction if n not in very_large_reduction]
        if long_reduction:
            _log_partition_separation(
                "ComboKernels: %d long reduction nodes are separated",
                long_reduction,
                not_reduction + short_reduction,
                node_info_map,
            )
        if very_large_reduction:
            log.debug(
                "ComboKernels: %d very large reduction nodes are separated",
                len(very_large_reduction),
            )
            nodes_per_ndim.extend([node] for node in very_large_reduction)
        large_pointwise = [
            n
            for n in not_reduction
            if not node_info_map[n].features.is_reduction()
            and len(node_info_map[n].tiling) == 2
            and V.graph.sizevars.optimization_hint(
                node_info_map[n].tiling["x"], fallback=1
            )
            > LARGE_NUMELS
        ]
        if large_pointwise:
            companion_nodes = [n for n in not_reduction if n not in large_pointwise]
            # TODO benchmark the performance when large pointwise nodes combining with others
            # Include the non-large pointwise companions because the diagnostic
            # describes a partition decision for the current candidate group.
            _log_partition_separation(
                "ComboKernels: %d large pointwise nodes are separated",
                large_pointwise,
                companion_nodes,
                node_info_map,
            )
            not_reduction = companion_nodes
            nodes_per_ndim.extend([node] for node in large_pointwise)

        nodes_per_ndim.extend(
            g for g in (not_reduction, short_reduction, long_reduction) if g
        )

    if sum(len(p) for p in nodes_per_ndim) != len(nodes):
        raise AssertionError("partitioned node count must equal input node count")
    return nodes_per_ndim


_custom_combo_kernel_horizontal_partition_algorithm: Callable[
    [
        list[BaseSchedulerNode],
        SIMDScheduling,
        dict[BaseSchedulerNode, NodeInfo],
    ],
    list[list[BaseSchedulerNode]],
] = _default_custom_combo_kernel_horizontal_partition


def set_custom_combo_kernel_horizontal_partition(
    algorithm: Callable[
        [
            list[BaseSchedulerNode],
            SIMDScheduling,
            dict[BaseSchedulerNode, NodeInfo],
        ],
        list[list[BaseSchedulerNode]],
    ],
) -> None:
    """Sets the algorithm used to partition nodes into horizontal partitions. Nodes in different partitions
    are implemented in different combo kernels. Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    The algorithm should take a list of nodes and return a list of list of nodes.

    The default algorithm is to partition nodes based on number of block dimensions.
    """
    global _custom_combo_kernel_horizontal_partition_algorithm
    _custom_combo_kernel_horizontal_partition_algorithm = algorithm


@dataclass
class PartitionState:
    partitions: list[list[BaseSchedulerNode]]
    cur_partition: list[BaseSchedulerNode]
    cur_count: int

    def finalize(self) -> None:
        if self.cur_partition:
            self.partitions.append(self.cur_partition)


@dataclass
class SubKernelSetup:
    uniquify_block_sizes: list[str]
    lhs_names: list[str]


@dataclass
class SubKernelCode:
    setup: IndentedBuffer
    body: IndentedBuffer
    setup_lhs_names: list[str]


@dataclass
class SharedBody:
    body: IndentedBuffer
    placeholder_names: list[str]
    args_by_subkernel: list[list[str]]
    setup_lhs_names: list[str]


@dataclass
class ComboLaunchConfig:
    kwargs: dict[str, int]
    num_warps: int
    num_stages: int


class ComboKernel(Kernel):
    """
    A kernel that combines multiple sub-kernels into a single fused kernel.
    """

    @staticmethod
    def _update_partition(
        partition_state: PartitionState,
        node_rw_count: int,
        node_info: BaseSchedulerNode,
    ) -> None:
        if partition_state.cur_count + node_rw_count > config.combo_kernel_max_num_args:
            partition_state.partitions.append(partition_state.cur_partition)
            partition_state.cur_partition = [node_info]
            partition_state.cur_count = node_rw_count
        else:
            partition_state.cur_count += node_rw_count
            partition_state.cur_partition.append(node_info)

    @staticmethod
    def _base_horizontal_partition(
        subkernel_nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        node_info_map: dict[BaseSchedulerNode, NodeInfo],
        custom_algorithm: bool,
    ) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        # TODO support combination of kernels with different block dimensions
        if len(subkernel_nodes) < 1:
            raise AssertionError(
                f"expected at least 1 subkernel node, got {len(subkernel_nodes)}"
            )
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_algorithm
        )

        ndim_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        yelem_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        all_partitions = []

        for node in subkernel_nodes:
            tiled_groups = node_info_map[node].tiling
            node_info = node

            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)

            ndim = len(tiled_groups)
            if ndim < 2:
                raise AssertionError(f"Combokernel not support tile {tiled_groups}")

            # Skip 2d reductions (r0_,r1_) and 3D pointwise (x,y,z) from combo
            keys = tiled_groups.keys()
            if ("r0_" in keys and "r1_" in keys) or "z" in keys:
                all_partitions.append([node_info])
                continue

            if not mixed_sizes and ndim == 3:
                y_elem = tiled_groups["y"]
                partition_state = yelem_to_partition_state[y_elem]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )
            else:
                if not (mixed_sizes or ndim <= 3):
                    raise AssertionError(f"No mixed sizes: tile {tiled_groups}")
                partition_state = ndim_to_partition_state[ndim]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )

        for partition_state in ndim_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)
        for partition_state in yelem_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)
        return all_partitions

    @staticmethod
    def horizontal_partition(
        nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        node_info_map: dict[BaseSchedulerNode, NodeInfo],
        custom_algorithm: bool = False,
    ) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnum)
        for each subkernel node where each sublist forms a ComboKernel. It horizontally partitions nodes into
        sublists in the following way:
            1) call _custom_combo_kernel_horizontal_partition_algorithm() if custom_algorithm is True
            2) then, call _base_horizontal_partition() to partition nodes into sublists, each sublist is
               guaranteed to not exceed CUDA limits for number of args (read/writes) and to have the same
               2D or 1D blocking strategy.
        """
        if custom_algorithm:
            raw_partitions = _custom_combo_kernel_horizontal_partition_algorithm(
                nodes, triton_scheduling, node_info_map
            )
        else:
            raw_partitions = [nodes]

        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        all_partitions = []
        for raw_partition in raw_partitions:
            all_partitions.extend(
                ComboKernel._base_horizontal_partition(
                    raw_partition, triton_scheduling, node_info_map, custom_algorithm
                )
            )
        return all_partitions

    class SequentialDispatch:
        """
        The dispatcher which dispatches the subkernels in a sequential manner:
        the blocks are first dispatched to the 1st subkernel (until it is filled),
        then to the 2nd subkernel, and so on.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """

        grid_expr = SequentialComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            if num == 0:
                cls._calculate_xblocks(kernel, code)
                code.splice(f"if pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice("pid_offset = pid")
            else:
                code.splice(f"elif pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice(f"pid_offset = pid - num_xblocks_{num - 1}")

        @classmethod
        def _calculate_xblocks(
            cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            x_numels_list = kernel.x_numels_list
            for i in range(len(x_numels_list)):
                xnumels, no_x_dim = (
                    (x_numels_list[i], False)
                    if isinstance(x_numels_list[i], str)
                    and cast(str, x_numels_list[i])[0] != "-"
                    or (
                        isinstance(x_numels_list[i], int)
                        and cast(int, x_numels_list[i]) > 0
                    )
                    else (kernel.min_x_blocks_list[i], True)
                )
                xblock_str = (
                    f"tl.cdiv({xnumels}, XBLOCK)" if not no_x_dim else f"{xnumels}"
                )
                if i == 0:
                    code.splice(f"num_xblocks_{i} = {xblock_str}")
                else:
                    code.splice(f"num_xblocks_{i} = num_xblocks_{i - 1} + {xblock_str}")

    class SequentialFlattenGridDispatch:
        """
        Flattened grid dispatch for per-subkernel blocks.
        Uses flattened grid (sum of x*y blocks, 1, 1) and computes
        x_pid_offset, y_pid_offset from the flattened pid.
        """

        grid_expr = SequentialFlattenComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            if num == 0:
                cls._calculate_total_blocks(kernel, code)
                code.splice(f"if pid < num_blocks_{num}:")
            else:
                code.splice(f"elif pid < num_blocks_{num}:")

            with code.indent():
                # Compute local pid within this subkernel's block range
                if num == 0:
                    code.splice("local_pid = pid")
                else:
                    code.splice(f"local_pid = pid - num_blocks_{num - 1}")

                # Compute x/y indices from flattened local_pid
                if kernel.y_tree_list[num]:
                    code.splice(f"x_pid_offset = local_pid % x_blocks_{num}")
                    code.splice(f"y_pid_offset = local_pid // x_blocks_{num}")
                else:
                    code.splice("x_pid_offset = local_pid")

        @classmethod
        def _calculate_total_blocks(
            cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            """
            Calculate total blocks for each subkernel (x_blocks * y_blocks)
            and cumulative block counts for dispatch boundaries.
            """
            for i, sub_kernel in enumerate(kernel.sub_kernels):
                no_x_dim = sub_kernel.no_x_dim
                xnumel = (
                    kernel.min_x_blocks_list[i] if no_x_dim else kernel.x_numels_list[i]
                )
                x_blocks_str = (
                    f"tl.cdiv({xnumel}, XBLOCK_{i})" if not no_x_dim else f"{xnumel}"
                )
                code.splice(f"x_blocks_{i} = {x_blocks_str}")

                if kernel.y_tree_list[i]:
                    numel = V.graph.sizevars.simplify(kernel.y_tree_list[i].numel)
                    ynumel = (
                        int(numel)
                        if isinstance(numel, (Integer, int))
                        else f"ynumel_{i}"
                    )
                    code.splice(f"y_blocks_{i} = tl.cdiv({ynumel}, YBLOCK_{i})")

                blocks_expr = (
                    f"x_blocks_{i} * y_blocks_{i}"
                    if kernel.y_tree_list[i]
                    else f"x_blocks_{i}"
                )
                code.splice(
                    f"num_blocks_{i} = {blocks_expr}"
                    if i == 0
                    else f"num_blocks_{i} = num_blocks_{i - 1} + {blocks_expr}"
                )

    class RoundRobinDispatch:
        """
        The dispatcher which dispatches the subkernels in a round robin manner:
        the blocks are interleavedly dispatched to each subkernel to execute them
        in parallel.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """

        grid_expr = RoundRobinComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            num_kernels = len(kernel.sub_kernels)
            if num == 0:
                cond = "if"
            else:
                cond = "elif"
            code.splice(f"{cond} pid % {num_kernels} == {num}:")
            with code.indent():
                code.splice(f"pid_offset = pid // {num_kernels}")

    class UniformDispatch:
        """
        Dispatch strategy for combo kernels where all sub-kernels have
        identical computation (same ops, same shapes, just different buffer
        pointers). Instead of if/elif branching, generates a single body
        with pointer-array indexing:
            kernel_idx = pid // num_blocks_per_kernel
            pid_offset = pid % num_blocks_per_kernel
            ptr = tl.load(ptr_array + kernel_idx).to(tl.pointer_type(dtype))
        This eliminates register pressure from duplicated code paths.
        """

        grid_expr = SequentialComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer  # noqa: F841
        ) -> None:
            # UniformDispatch does not use pid_range codegen;
            # the uniform path in codegen_kernel() handles dispatch directly.
            pass

    def __init__(
        self,
        triton_kernel_cls: type[TritonKernel],
        enable_autotune: bool = False,
        mixed_sizes: bool = False,
        per_subkernel_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.triton_kernel_cls = triton_kernel_cls
        self.sub_kernels: list[TritonKernel] = []
        self.iter_vars_count = itertools.count()
        self.grids: list[list[int]] = []
        self.min_x_blocks_list: list[int | str] = []
        self.x_numels_list: list[int | str] = []
        self.y_tree_list: list = []
        self.enable_autotune = enable_autotune
        self.mixed_sizes = mixed_sizes
        self.per_subkernel_blocks = per_subkernel_blocks
        self.dispatch_class: (
            type[
                ComboKernel.SequentialDispatch
                | ComboKernel.SequentialFlattenGridDispatch
                | ComboKernel.RoundRobinDispatch
                | ComboKernel.UniformDispatch
            ]
            | None
        ) = None
        self.block_args: list[str] = []
        # the following are used when autotuning is disabled
        self.block_size_1d = DEFAULT_COMBO_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_COMBO_BLOCK_SIZE_2D
        self.num_warps = 8
        self.block_size_reduce = 256
        self.dynamic_shape_args: list[str] = []
        self.no_bench_stitched_config: triton.Config | None = None
        self.combo_compile_time_autotune = False
        # Compile-time autotune: per-subkernel winning block sizes (XBLOCK_0, ...), passed as args.
        self.stitched_block_config: dict[str, int] | None = None
        # Distinct winner launch configs across the subkernels; seeds the combo's kernel-level autotune.
        self.combo_launch_candidates: list[ComboLaunchConfig] = []
        self._uniform_dispatch_info: dict[str, Any] | None = None

    @property
    def bake_blocks(self) -> bool:
        """Bake block sizes as constexpr in the body (only the no-autotune path, which has no
        autotuner to pass a config through). Otherwise blocks are args; compile-time autotune
        supplies the chosen blocks via the config (default_config)."""
        return not self.enable_autotune

    def create_sub_kernel(self, triton_kernel: TritonKernel) -> TritonKernel:
        sub_kernel = triton_kernel
        # pyrefly: ignore [bad-assignment]
        metrics.generated_kernel_count -= 1
        sub_kernel.args = self.args
        sub_kernel.iter_vars_count = self.iter_vars_count
        sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        self.sub_kernels.append(sub_kernel)
        return sub_kernel

    @staticmethod
    def create_triton_kernel(
        tiling: dict[str, sympy.Expr],
        features: SIMDKernelFeatures,
        optimize_mask: bool,
        triton_kernel_cls: type[TritonKernel],
        tiling_scores: dict[str, sympy.Expr] | None = None,
        per_subkernel_blocks: bool = False,
    ) -> TritonKernel:
        """
        Only allow optimize_mask=True when 1) sequential dispatch is used,
        2) numels except x dimension are the same for each sub kernel.
        """
        # Flattened dispatch: all dimensions derived from single pid
        if per_subkernel_blocks:
            pid_cache = {
                "tl.program_id(0)": "x_pid_offset",
                "tl.program_id(1)": "y_pid_offset",
            }
        else:
            pid_cache = {"tl.program_id(0)": "pid_offset"}

        kwargs: dict[str, Any] = dict(
            pid_cache=pid_cache,
            optimize_mask=optimize_mask,
            is_combo_kernel=True,
            per_subkernel_blocks=per_subkernel_blocks,
            # foreach kernels don't work with cooperative reductions
            override_cooperative_reduction=False,
            tiling_scores=tiling_scores,
        )
        triton_kernel_cls.apply_feature_required_overrides(features, kwargs)

        return triton_kernel_cls(tiling, features=features, **kwargs)

    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> SubKernelSetup:
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        grid = []
        lhs_names: list[str] = []
        uniquify_block_sizes = []
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if isinstance(simplified_tree_numel, (Integer, int)):
                lhs_name = f"{tree.prefix}numel"
                code.writeline(f"{lhs_name} = {int(simplified_tree_numel)}")
                lhs_names.append(lhs_name)
            else:
                if f"{tree.prefix}numel_{num}" not in self.dynamic_shape_args:
                    raise AssertionError(
                        f"{tree.prefix}numel_{num} not in dynamic_shape_args"
                    )
                uniquify_block_sizes.append(f"{tree.prefix}numel")

            if not tree.is_reduction:
                if isinstance(simplified_tree_numel, (Integer, int)):
                    grid.append(int(simplified_tree_numel))
                else:
                    # pyrefly: ignore [bad-argument-type]
                    grid.append(f"{tree.prefix}numel_{num}")

            if tree.is_reduction and sub_kernel.persistent_reduction:
                val = TritonKernel._get_persistent_RBLOCK(tree.numel)
                lhs_names.append(f"{tree.prefix.upper()}BLOCK_{num}")
                code.writeline(
                    f"{tree.prefix.upper()}BLOCK_{num}: tl.constexpr = {val}"
                )

            if tree.prefix == "x" and sub_kernel.no_x_dim:
                lhs_names.append(f"XBLOCK_{num}")
                code.writeline(f"XBLOCK_{num}: tl.constexpr = 1")
                uniquify_block_sizes.append("XBLOCK")
            elif tree.prefix in ("x", "y") and self.per_subkernel_blocks:
                uniquify_block_sizes.append(f"{tree.prefix.upper()}BLOCK")
            elif tree.is_reduction:
                if self.per_subkernel_blocks or sub_kernel.persistent_reduction:
                    uniquify_block_sizes.append(f"{tree.prefix.upper()}BLOCK")
        self.grids.append(grid)
        return SubKernelSetup(
            uniquify_block_sizes=uniquify_block_sizes,
            lhs_names=lhs_names,
        )

    def min_x_blocks_sub_kernel(self, sub_kernel: TritonKernel, num: int) -> None:
        """
        Kernels with no_x_dim being true has no tunable XBLOCK. They have a fixed number of X blocks.
        Grid calculation needs to make sure that they are assigned with enough number of blocks.
        """
        min_x_blocks: int | str = 0
        x_numels: int | str = 0
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if tree.prefix == "x":
                if isinstance(simplified_tree_numel, (Integer, int)):
                    x_numels = int(simplified_tree_numel)
                else:
                    x_numels = f"{tree.prefix}numel_{num}"
                if sub_kernel.no_x_dim:
                    min_x_blocks = x_numels
                    x_numels = (
                        # pyrefly: ignore [unsupported-operation]
                        -min_x_blocks
                        if isinstance(x_numels, int)
                        # pyrefly: ignore [redundant-cast]
                        else "-" + cast(str, x_numels)
                    )
                else:
                    if isinstance(simplified_tree_numel, (Integer, int)):
                        x_numels = int(simplified_tree_numel)
                    else:
                        x_numels = f"{tree.prefix}numel_{num}"
        self.min_x_blocks_list.append(min_x_blocks)
        self.x_numels_list.append(x_numels)

    def select_heuristics(self, sub_kernel: TritonKernel) -> tuple[str, dict[str, int]]:
        size_hints = {
            prefix: next_power_of_2(V.graph.sizevars.optimization_hint(numel))
            for prefix, numel in sub_kernel.numels.items()
            if not prefix_is_reduction(prefix) or sub_kernel.inside_reduction
        }
        if sub_kernel.persistent_reduction:
            if not sub_kernel.inside_reduction:
                raise AssertionError(
                    "persistent_reduction sub_kernel must be inside_reduction"
                )
            heuristics = "persistent_reduction"
        elif sub_kernel.inside_reduction:
            heuristics = "reduction"
        else:
            heuristics = "pointwise"
        return heuristics, size_hints

    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]:
        if not self.enable_autotune and self.no_bench_stitched_config is None:
            return "foreach", size_hints_list[0], self.sub_kernels[0]
        if "reduction" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1]["x"] if heuristics_list[x[0]] == "reduction" else 0,
            )
            return heuristics_list[i], size_hints_list[i], self.sub_kernels[i]
        elif "pointwise" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1]["x"] if heuristics_list[x[0]] == "pointwise" else 0,
            )
            # modify size_hint to avoid oom check fail (may be a false alarm)
            num_pointwise = len([e for e in heuristics_list if e == "pointwise"])
            num_reduction = len([e for e in heuristics_list if e == "reduction"])
            num_persistent_reduction = len(
                [e for e in heuristics_list if e == "persistent_reduction"]
            )
            if num_reduction != 0:
                raise AssertionError(
                    "combining pointwise and reduction are not supported yet."
                )
            heuristics = (
                "pointwise_with_reduction"
                if num_persistent_reduction > 0
                else "pointwise"
            )
            if len(heuristics_list) - num_pointwise >= 4:
                size_hints = size_hints_list[i]
                size_hints["x"] = min(128, size_hints["x"])
            return heuristics, size_hints_list[i], self.sub_kernels[i]
        else:
            # find persistent_reduction with maximum rnumel
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: max(
                    (v for k, v in x[1].items() if prefix_is_reduction(k))
                ),
            )
            return heuristics_list[i], size_hints_list[i], self.sub_kernels[i]

    def get_mutated_args_sub_kernels(self) -> list[str]:
        mutated_args: OrderedSet[str] = OrderedSet()
        for sub_kernel in self.sub_kernels:
            for mutation in sub_kernel.mutations:
                if mutation in sub_kernel.args.input_buffers:
                    mutated_args.add(sub_kernel.args.input_buffers[mutation])
                if (
                    mutation in sub_kernel.args.inplace_buffers
                    and mutation not in V.graph.removed_buffers
                    and mutation not in sub_kernel.removed_buffers
                ):
                    mutated_args.add(
                        cast(
                            InplacedBuffer, sub_kernel.args.inplace_buffers[mutation]
                        ).inner_name
                    )
                if mutation in sub_kernel.args.output_buffers:
                    arg = sub_kernel.args.output_buffers[mutation]
                    if isinstance(arg, RemovedArg):
                        raise AssertionError("mutated output buffer arg was removed")
                    mutated_args.add(arg)
        return sorted(mutated_args)

    def _detect_uniform_subkernels(self) -> bool:
        """
        Detect whether all sub-kernels are structurally identical (same ops,
        same shapes, differing only in buffer pointers). If so, store the
        mapping info in self._uniform_dispatch_info and return True.

        This performs a structural check without generating bodies:
        - All sub-kernels must have the same range tree structure
        - All sub-kernels must have the same reduction/no_x_dim flags
        - Must have at least 2 sub-kernels
        - No dynamic shapes (simplifies the initial implementation)
        - No per-subkernel blocks mode
        """
        if len(self.sub_kernels) < 2:
            log.debug("uniform dispatch: skipped (< 2 sub-kernels)")
            return False

        # Cost gate: uniform dispatch's benefit (one shared body instead of an
        # N-way body/branch chain) scales with group size N, while its
        # pointer-table build/copy cost is roughly fixed. Small groups pay the
        # overhead for little gain (N=2 is ~half of dashboard groups and net
        # neutral/negative), so require a minimum group size. getattr with a
        # default keeps this working even where config.py predates the knob (e.g.
        # a branch that ships only the codegen change).
        min_kernels = max(
            2, getattr(config, "combo_kernel_uniform_dispatch_min_kernels", 32)
        )
        if len(self.sub_kernels) < min_kernels:
            log.debug(
                "uniform dispatch: skipped (group size %d < min_kernels %d)",
                len(self.sub_kernels),
                min_kernels,
            )
            return False

        # cudagraphs is supported: the pointer table is built from a persistent
        # pinned staging buffer (valid, stable source for the captured H2D on
        # replay) into a cudagraph-pool-tracked device table, with the event guard
        # skipped during capture. See ComboKernel._emit_uniform_pointer_tables /
        # the _uniform_stage wrapper helper.

        # Bail on per-subkernel blocks
        if config.combo_kernel_per_subkernel_blocks:
            log.debug("uniform dispatch: skipped (per_subkernel_blocks)")
            return False

        # Check for dynamic shapes directly (can't rely on dynamic_shape_args
        # as it may not be populated yet at detection time)
        for sub in self.sub_kernels:
            for tree in sub.range_trees:
                if not isinstance(
                    V.graph.sizevars.simplify(tree.numel), (Integer, int)
                ):
                    log.debug("uniform dispatch: skipped (dynamic shape in %s)", tree.prefix)
                    return False

        ref = self.sub_kernels[0]
        ref_trees = [(t.prefix, t.is_reduction, V.graph.sizevars.simplify(t.numel))
                     for t in ref.range_trees]

        for idx, sub in enumerate(self.sub_kernels[1:], 1):
            # Must have same flags
            if sub.no_x_dim != ref.no_x_dim:
                log.debug("uniform dispatch: skipped (no_x_dim mismatch at sub %d)", idx)
                return False
            if sub.inside_reduction != ref.inside_reduction:
                log.debug("uniform dispatch: skipped (inside_reduction mismatch at sub %d)", idx)
                return False
            if sub.persistent_reduction != ref.persistent_reduction:
                log.debug("uniform dispatch: skipped (persistent_reduction mismatch at sub %d)", idx)
                return False

            # Must have same range tree structure
            sub_trees = [(t.prefix, t.is_reduction, V.graph.sizevars.simplify(t.numel))
                         for t in sub.range_trees]
            if sub_trees != ref_trees:
                log.debug("uniform dispatch: skipped (range tree mismatch at sub %d: %s vs %s)", idx, sub_trees, ref_trees)
                return False

        # All structural checks passed
        log.debug("uniform dispatch: structural check PASSED for %d sub-kernels", len(self.sub_kernels))
        self._uniform_dispatch_info = {}
        return True

    def _build_uniform_dispatch_info(self) -> bool:
        """
        After bodies are generated, verify all sub-kernels share an identical
        structured op trace and build the slot mapping for pointer arrays.

        The op trace is recorded during body codegen (see
        ``TritonKernel.record_op_trace``) with buffer names, CSE temporaries and
        range symbols normalized to positional placeholders.  Structural
        equality of the traces therefore means the sub-kernel bodies are
        identical up to their buffer pointers -- exactly the condition uniform
        dispatch requires -- without depending on the textual form of the
        generated Triton code.

        Returns True if uniform dispatch can proceed, False to fall back.
        """
        first_trace = self.sub_kernels[0].op_trace
        if not first_trace:
            log.debug("uniform dispatch build: FAILED - empty op trace")
            return False
        if any(sub.op_trace != first_trace for sub in self.sub_kernels[1:]):
            log.debug("uniform dispatch build: FAILED - op trace mismatch")
            return False

        # Resolve inner buffer arg name -> outer call arg + dtype from the
        # shared combo-kernel args namespace.
        argdefs, call_args, precompile_args, _ = self.args.python_argdefs()
        inner_to_outer: dict[str, str] = {}
        inner_to_dtype: dict[str, Any] = {}
        for argdef, call_arg, precompile_arg in zip(argdefs, call_args, precompile_args):
            inner_to_outer[argdef.name] = call_arg
            if hasattr(precompile_arg, "dtype"):
                inner_to_dtype[argdef.name] = precompile_arg.dtype

        # Per sub-kernel, the ordered pointer args referenced by the body taken
        # from the structured trace (first-appearance order).  Filtering to
        # real kernel args drops removed/intermediate buffers that never appear
        # in the emitted body.
        buf_refs_per_kernel: list[list[str]] = [
            [name for name in sub.op_trace_buffer_arg_names if name in inner_to_outer]
            for sub in self.sub_kernels
        ]

        ref_count = len(buf_refs_per_kernel[0])
        if ref_count == 0:
            log.debug("uniform dispatch build: FAILED - no buffer refs in body")
            return False
        if any(len(refs) != ref_count for refs in buf_refs_per_kernel[1:]):
            log.debug("uniform dispatch build: FAILED - buffer ref count mismatch")
            return False

        # Correctness guard: every pointer/buffer kernel arg must be reachable
        # through the op trace (i.e. slotted per sub-kernel). A buffer that is
        # read only via an indirect (gathered) index bypasses record_op_trace --
        # CSEProxy.load returns early to indirect_load before recording -- so it
        # is neither compared for sub-kernel equality nor turned into a per-slot
        # pointer table. Such a buffer is appended to the uniform call once and
        # shared across every group, so the single generated body (built from
        # sub-kernel 0) hardcodes group 0's buffer while the other groups'
        # buffers become dead args: all non-first groups silently read group 0's
        # data (observed on Super_SloMo / inception_v3 upsample_bilinear2d, where
        # group 1 reused group 0's input). Fall back to sequential dispatch
        # whenever any buffer arg is unslotted.
        slotted_inner_names = {
            name for refs in buf_refs_per_kernel for name in refs
        }
        unslotted_buffers = set(inner_to_dtype) - slotted_inner_names
        if unslotted_buffers:
            log.debug(
                "uniform dispatch build: FAILED - unslotted buffer arg(s) %s "
                "(likely reached via indirect load); falling back to sequential",
                sorted(unslotted_buffers),
            )
            return False

        # Build slot info: for each slot position, record the buffer inner name
        # used in sub-kernel 0's body plus the outer (call_arg) names and dtype.
        # Op-trace equality guarantees the buffers correspond positionally
        # across sub-kernels.
        slots: list[dict[str, Any]] = []
        ref_buf_refs = buf_refs_per_kernel[0]
        for slot_idx in range(ref_count):
            slot_call_args = []
            slot_dtype = None
            for refs in buf_refs_per_kernel:
                inner_name = refs[slot_idx]
                outer = inner_to_outer.get(inner_name)
                if outer is None:
                    return False  # can't resolve buffer
                slot_call_args.append(outer)
                if slot_dtype is None:
                    slot_dtype = inner_to_dtype.get(inner_name)

            slots.append({
                "inner_name": ref_buf_refs[slot_idx],  # name used in the body
                "call_args": slot_call_args,  # outer buffer names per sub-kernel
                "dtype": slot_dtype,  # torch dtype for pointer cast
            })

        self._uniform_dispatch_info = {
            "slots": slots,
            "body": self.sub_kernels[0].body,
            "buf_refs": buf_refs_per_kernel,
        }
        return True

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is not None:
            return
        if config.combo_kernel_uniform_dispatch and self._detect_uniform_subkernels():
            self.dispatch_class = ComboKernel.UniformDispatch
            return
        if self.per_subkernel_blocks:
            self.dispatch_class = ComboKernel.SequentialFlattenGridDispatch
            return
        # mixed_sizes is used for optimize_mask, so it only allows sequential dispatch
        # Not mixed sizes on y dim technically is ok to use round robin as wells.
        if not self.mixed_sizes or any(isinstance(e, str) for e in self.x_numels_list):
            # str in x_numels_list means a dynamic shape
            self.dispatch_class = ComboKernel.SequentialDispatch
            return
        # A negative x_blocks_list element means the kernel is not tunable,
        # i.e., no_x_dim = True
        x_numels_list = [abs(cast(int, e)) for e in self.x_numels_list]
        total = max(x_numels_list) * len(x_numels_list)
        needed = sum(x_numels_list)
        if needed / total > BLOCK_UTILIZATION:
            # Introduced overhead (masked blocks) is less than 20%
            self.dispatch_class = ComboKernel.RoundRobinDispatch
        else:
            self.dispatch_class = ComboKernel.SequentialDispatch

    def jit_line(
        self,
        heuristics: str,
        size_hints: dict[str, int],
        selected_kernel: TritonKernel,
        signature: list[Any],
        argdefs: list[ArgName],
        size_hints_list: list[dict[str, int]],
        pointwise_with_reduce: bool = False,
    ) -> str:
        """Write the @triton_heuristics.<heuristics> decorator line for the combo kernel."""

        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        sig_meta = signature_to_meta(
            signature, size_dtype=size_dtype, argdefs=argdefs
        )
        # Slot pointer-table args are int64 tensors of GPU addresses but modeled
        # as SizeArg for config_of compatibility.  Override their signature type
        # from scalar (i32/i64) to pointer (*i64) so Triton treats them correctly.
        for argdef in argdefs:
            if argdef.name.startswith("_slot_") and argdef.name.endswith("_ptrs"):
                sig_meta[argdef.name] = "*i64"
        triton_meta: TritonMeta = cast(
            TritonMeta,
            {
                "signature": sig_meta,
                "device": DeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
                "constants": {},
                # Inherit enable_fp_fusion, launch_pdl, disable_ftz so combo kernels
                # compile with the same Triton options as standalone kernels.
                **TritonKernel.triton_meta_common(),
            },
        )

        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        triton_meta["configs"] = [
            config_of(signature, skip_cpp_wrapper_input_tensor_alignment=True)
        ]

        mutated_args = self.get_mutated_args_sub_kernels()
        dispatch = self.dispatch_class
        if dispatch is None:
            raise AssertionError("dispatch_class must not be None")

        # Compute the max persistent R0_BLOCK across sub-kernels.
        # This is used by _reduction_configs() to avoid generating configs
        # where XBLOCK * max_persistent_rblock creates pathologically large
        # tiles that cause extreme ROCm compilation times.
        # The max_persistent_rblock mirrors how R0_BLOCK is computed in
        # codegen_static_numels_sub_kernel() for persistent reductions.
        max_persistent_rblock = 0
        if not self.per_subkernel_blocks:
            max_persistent_rblock = max(
                (
                    TritonKernel._get_persistent_RBLOCK(tree.numel)
                    for sub in self.sub_kernels
                    if sub.persistent_reduction
                    for tree in sub.range_trees
                    if tree.is_reduction
                ),
                default=0,
            )

        inductor_meta = {
            "grid_type": dispatch.grid_expr.__name__,
            "combo_grid_meta": self.combo_grid_meta(size_hints_list),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            # Matches triton.py:codegen_kernel(): inference/backward graphs skip
            # CPU-copy of mutated args during autotune retries; training-forward
            # graphs must keep it to preserve benchmark inputs across retries.
            "optimize_mem": V.graph.is_inference or V.graph.is_backward,
            **self.triton_kernel_cls.inductor_meta_common(),
        }
        if max_persistent_rblock > 0:
            inductor_meta["max_persistent_rblock"] = max_persistent_rblock

        # Sum per-sub-kernel bandwidth / FLOP estimates for the combo launch.
        sub_metas = [sub.inductor_meta_per_kernel() for sub in self.sub_kernels]
        self._kernel_num_gb = sum(m.get("kernel_num_gb") or 0 for m in sub_metas)
        if config.benchmark_kernel or config.profile_bandwidth:
            inductor_meta["kernel_num_gb"] = self._kernel_num_gb
        if config.benchmark_kernel:
            inductor_meta["kernel_flop"] = sum(
                m.get("kernel_flop") or 0 for m in sub_metas
            )

        sub_kernel = selected_kernel
        if heuristics == "foreach":
            heuristics_line = f"""
                @triton_heuristics.foreach(
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                )
                @triton.jit
            """
        elif sub_kernel.inside_reduction:
            reduction_hint = sub_kernel.features.get_reduction_hint(
                sub_kernel.tiling_scores
            )
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                tile_hint = "tile_hint=TileHint.SQUARE,"
            else:
                tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """

        self.triton_meta = triton_meta
        self.inductor_meta = inductor_meta

        return heuristics_line

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        has_yblock = any(self.y_tree_list)
        stitched_kwargs = (
            self.no_bench_stitched_config.kwargs
            if self.no_bench_stitched_config is not None
            else None
        )

        for block in self.block_args:
            if stitched_kwargs is not None and block in stitched_kwargs:
                size = stitched_kwargs[block]
            elif "YBLOCK" in block:
                size = self.block_size_2d
            elif "XBLOCK" in block:
                size = self.block_size_2d if has_yblock else self.block_size_1d
            elif "R0_BLOCK" in block:
                size = self.block_size_reduce
            else:
                raise AssertionError(f"{block} is not supported without autotuning")
            code.splice(f"{block}: tl.constexpr = {size}")

    def get_block_args(self) -> list[ConstexprArg]:
        """
        Calculate blocks from sub_kernels and range_trees.
        Update self.block_args, self.y_tree_list
        Return the block args
        """
        block_names = {}
        for i, sub_kernel in enumerate(self.sub_kernels):
            y_tree = None
            for tree in sub_kernel.range_trees:
                if tree.is_reduction and (
                    not sub_kernel.inside_reduction or sub_kernel.persistent_reduction
                ):
                    continue
                if tree.prefix == "x" and sub_kernel.no_x_dim:
                    continue
                if tree.prefix == "y":
                    y_tree = tree
                if self.per_subkernel_blocks:
                    block_names[f"{tree.prefix.upper()}BLOCK_{i}"] = tree.prefix
                else:
                    block_names[f"{tree.prefix.upper()}BLOCK"] = tree.prefix
            self.y_tree_list.append(y_tree)
        self.block_args = list(block_names.keys())

        return [ConstexprArg(x) for x in block_names]

    def add_numel_to_args(
        self, argdefs: list[ArgName], signature: list[Any]
    ) -> list[ArgName]:
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.active_range_trees():
                if not isinstance(tree.numel, (Integer, int)):
                    # only if it is a dynamic shape
                    sizearg = SizeArg(f"{tree.prefix}numel_{num}", tree.numel)
                    signature.append(sizearg)
                    argdefs.append(ArgName(f"{tree.prefix}numel_{num}"))
                    self.dynamic_shape_args.append(f"{tree.prefix}numel_{num}")
        return argdefs

    def add_numel_to_call_args(
        self, name: str, call_args: list[Any], arg_types: list[Any]
    ) -> None:
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.range_trees:
                numel_name = f"{tree.prefix}numel_{num}"
                if numel_name not in self.dynamic_shape_args:
                    continue
                if isinstance(tree.numel, (Integer, Symbol)):
                    expr = tree.numel
                else:
                    expr = V.graph.wrapper_code.generate_numel_expr(
                        name, tree, suffix=str(num)
                    )

                if not tree.is_reduction or sub_kernel.inside_reduction:
                    call_args.append(expr)
                    arg_types.append(type(expr))

    def kernel_benchmark_extra_args(self) -> list[str]:
        extra_args = []
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.range_trees:
                numel_name = f"{tree.prefix}numel_{num}"
                if numel_name not in self.dynamic_shape_args:
                    continue

                if not tree.is_reduction or sub_kernel.inside_reduction:
                    extra_args.append(
                        str(V.graph.sizevars.optimization_hint(tree.numel))
                    )
        return extra_args

    def _can_share_body(
        self,
        heuristics_list: list[str],
    ) -> bool:
        if len(self.sub_kernels) < 2:
            return False
        if self.enable_autotune or self.per_subkernel_blocks:
            return False
        if torch.version.hip is not None:
            # The shared-body form joins live pointer placeholders after a
            # many-way dispatch branch. HIP/Triton currently has pathological
            # compile times for that IR shape on large foreach lists, so keep
            # ROCm on the existing per-branch body emission path.
            return False
        if self.dispatch_class not in (
            ComboKernel.SequentialDispatch,
            ComboKernel.RoundRobinDispatch,
        ):
            return False
        if self.dynamic_shape_args or any(self.y_tree_list):
            return False
        if any(
            sub_kernel.no_x_dim
            or sub_kernel.inside_reduction
            or sub_kernel.persistent_reduction
            for sub_kernel in self.sub_kernels
        ):
            return False
        return all(heuristic == "pointwise" for heuristic in heuristics_list)

    @staticmethod
    def _plain_lines(code: IndentedBuffer) -> list[str] | None:
        lines: list[str] = []
        for line in code._lines:
            if isinstance(line, str):
                lines.append(line)
            elif isinstance(line, DeferredLineBase):
                evaluated = line()
                if evaluated is not None:
                    lines.append(evaluated)
            else:
                return None
        return lines

    @staticmethod
    def _replace_names_in_line(line: str, replacements: dict[str, str]) -> str | None:
        if not replacements:
            return line
        try:
            pieces: list[str] = []
            cursor = 0
            for token in tokenize.generate_tokens(io.StringIO(line).readline):
                if token.type == tokenize.NAME and token.string in replacements:
                    start, end = token.start[1], token.end[1]
                    pieces.append(line[cursor:start])
                    pieces.append(replacements[token.string])
                    cursor = end
            pieces.append(line[cursor:])
            return "".join(pieces)
        except tokenize.TokenError:
            return None

    @classmethod
    def _replace_names(
        cls, lines: list[str], replacements: dict[str, str]
    ) -> list[str] | None:
        replaced: list[str] = []
        for line in lines:
            new_line = cls._replace_names_in_line(line, replacements)
            if new_line is None:
                return None
            replaced.append(new_line)
        return replaced

    @staticmethod
    def _names_in_lines(lines: list[str]) -> OrderedSet[str] | None:
        names: OrderedSet[str] = OrderedSet()
        try:
            for line in lines:
                for token in tokenize.generate_tokens(io.StringIO(line).readline):
                    if token.type == tokenize.NAME:
                        names.add(token.string)
        except tokenize.TokenError:
            return None
        return names

    @staticmethod
    def _range_tree_names(sub_kernel: TritonKernel) -> list[str]:
        names: list[str] = []
        for tree in sub_kernel.range_trees:
            names.append(tree.name)
            names.extend(entry.name for entry in tree.nodes.values())
        return names

    @classmethod
    def _range_tree_name_replacements(
        cls, first: TritonKernel, sub_kernel: TritonKernel
    ) -> dict[str, str] | None:
        if len(first.range_trees) != len(sub_kernel.range_trees):
            return None

        for first_tree, tree in zip(
            first.range_trees, sub_kernel.range_trees, strict=True
        ):
            if (
                first_tree.prefix != tree.prefix
                or first_tree.tensor_dim != tree.tensor_dim
                or first_tree.is_reduction != tree.is_reduction
            ):
                return None

        first_names = cls._range_tree_names(first)
        names = cls._range_tree_names(sub_kernel)
        if len(first_names) != len(names):
            return None
        return {
            name: first_name
            for first_name, name in zip(first_names, names, strict=True)
            if name != first_name
        }

    @classmethod
    def _body_name_replacements(
        cls,
        first: TritonKernel,
        sub_kernel: TritonKernel,
        args: list[str],
    ) -> dict[str, str] | None:
        first_cse_names = {
            canonical: name for name, canonical in first._op_trace_cse_names.items()
        }
        replacements: dict[str, str] = {}
        for name, canonical in sub_kernel._op_trace_cse_names.items():
            first_name = first_cse_names.get(canonical)
            if first_name is None:
                return None
            if name != first_name:
                replacements[name] = first_name

        range_replacements = cls._range_tree_name_replacements(first, sub_kernel)
        if range_replacements is None:
            return None
        replacements.update(range_replacements)

        for i, arg in enumerate(args):
            replacements[arg] = f"foreach_arg{i}"
        return replacements

    @staticmethod
    def _compatible_shared_arg_properties(
        args_by_subkernel: list[list[str]],
        tensor_args: dict[str, TensorArg],
    ) -> bool:
        if not args_by_subkernel:
            return False
        num_args = len(args_by_subkernel[0])
        if num_args == 0:
            return False
        if any(len(args) != num_args for args in args_by_subkernel):
            return False
        if any(arg not in tensor_args for args in args_by_subkernel for arg in args):
            return False

        for arg_index in range(num_args):
            properties = OrderedSet(
                [
                    (
                        tensor_args[args[arg_index]].dtype,
                        is_unaligned_buffer(tensor_args[args[arg_index]]),
                    )
                    for args in args_by_subkernel
                ]
            )
            if len(properties) != 1:
                return False

        return True

    @classmethod
    def _setup_lhs_names(cls, sub_kernel_codes: list[SubKernelCode]) -> list[str]:
        names: list[str] = []
        seen: OrderedSet[str] = OrderedSet()
        for sub_kernel_code in sub_kernel_codes:
            for name in sub_kernel_code.setup_lhs_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def _try_get_shared_body(
        self,
        sub_kernel_codes: list[SubKernelCode],
        signature: list[Any],
        heuristics_list: list[str],
    ) -> SharedBody | None:
        if not self._can_share_body(heuristics_list):
            return None

        tensor_args = {arg.name: arg for arg in signature if isinstance(arg, TensorArg)}
        if not tensor_args:
            return None
        first_sub_kernel = self.sub_kernels[0]
        first_trace = first_sub_kernel.op_trace
        if not first_trace:
            return None
        if any(
            sub_kernel.op_trace != first_trace for sub_kernel in self.sub_kernels[1:]
        ):
            return None

        transformed_bodies: list[list[str]] = []
        args_by_subkernel: list[list[str]] = []

        for sub_kernel, sub_kernel_code in zip(
            self.sub_kernels, sub_kernel_codes, strict=True
        ):
            lines = self._plain_lines(sub_kernel_code.body)
            if lines is None:
                return None
            body_names = self._names_in_lines(lines)
            if body_names is None:
                return None
            if any(
                arg in body_names and arg not in tensor_args
                for arg in sub_kernel.op_trace_buffer_arg_names
            ):
                return None
            args = [
                arg
                for arg in sub_kernel.op_trace_buffer_arg_names
                if arg in tensor_args and arg in body_names
            ]
            # Every live tensor pointer in the emitted body must come from the
            # structured trace so placeholder substitution cannot miss it.
            if any(arg in body_names and arg not in args for arg in tensor_args):
                return None
            args_by_subkernel.append(args)

            replacements = self._body_name_replacements(
                first_sub_kernel, sub_kernel, args
            )
            if replacements is None:
                return None
            transformed = self._replace_names(lines, replacements)
            if transformed is None:
                return None

            transformed_bodies.append(transformed)

        if any(body != transformed_bodies[0] for body in transformed_bodies[1:]):
            return None
        if not self._compatible_shared_arg_properties(args_by_subkernel, tensor_args):
            return None

        setup_lhs_names = self._setup_lhs_names(sub_kernel_codes)

        body = IndentedBuffer()
        body.writelines(transformed_bodies[0])
        return SharedBody(
            body=body,
            placeholder_names=[
                f"foreach_arg{i}" for i in range(len(args_by_subkernel[0]))
            ],
            args_by_subkernel=args_by_subkernel,
            setup_lhs_names=setup_lhs_names,
        )

    def _codegen_sub_kernel_bodies(
        self,
    ) -> list[SubKernelCode]:
        sub_kernel_codes: list[SubKernelCode] = []
        for num, sub_kernel in enumerate(self.sub_kernels):
            setup = IndentedBuffer()
            sub_kernel_setup = self.codegen_static_numels_sub_kernel(
                setup, sub_kernel, num
            )
            sub_kernel.codegen_prologue(sub_kernel.body)
            sub_kernel.codegen_body()
            sub_kernel._filter_pdl(sub_kernel.body)
            body = self.uniquify_block_sizes(
                sub_kernel.body, num, sub_kernel_setup.uniquify_block_sizes
            )
            sub_kernel_codes.append(
                SubKernelCode(
                    setup=setup,
                    body=body,
                    setup_lhs_names=sub_kernel_setup.lhs_names,
                )
            )
        return sub_kernel_codes

    def _codegen_branch(
        self,
        code: IndentedBuffer,
        num: int,
        sub_kernel_code: SubKernelCode,
    ) -> None:
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")
        self.dispatch_class.codegen_pid_range(self, num, code)
        with code.indent():
            code.splice(sub_kernel_code.setup)
            code.splice(sub_kernel_code.body)

    def _codegen_shared_branches(
        self,
        code: IndentedBuffer,
        sub_kernel_codes: list[SubKernelCode],
        shared_body: SharedBody,
    ) -> None:
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")
        for num, sub_kernel_code in enumerate(sub_kernel_codes):
            self.dispatch_class.codegen_pid_range(self, num, code)
            with code.indent():
                code.splice(sub_kernel_code.setup)
                for placeholder, arg in zip(
                    shared_body.placeholder_names,
                    shared_body.args_by_subkernel[num],
                    strict=True,
                ):
                    code.writeline(f"{placeholder} = {arg}")

        code.splice("else:")
        with code.indent():
            code.splice("pid_offset = 0")
            for name in shared_body.setup_lhs_names:
                code.writeline(f"{name} = 0")
            for placeholder, arg in zip(
                shared_body.placeholder_names,
                shared_body.args_by_subkernel[0],
                strict=True,
            ):
                code.writeline(f"{placeholder} = {arg}")

        code.splice(shared_body.body)

    def codegen_kernel(self, name: str | None = None) -> str:
        """Generate the triton code for a combo kernel that fuses multiple sub-kernels."""
        # TODO: is it correct to use the first sub kernel's heuristics?
        heuristics_list, size_hints_list = [], []
        for subkernel in self.sub_kernels:
            h, s = self.select_heuristics(subkernel)
            heuristics_list.append(h)
            size_hints_list.append(s)
        heuristics, size_hints, selected_kernel = self.select_combo_heuristics(
            heuristics_list, size_hints_list
        )
        pointwise_with_reduction, heuristics = (
            (True, "pointwise")
            if heuristics == "pointwise_with_reduction"
            else (False, heuristics)
        )

        # Early detection for uniform dispatch (before jit_line/select_dispatch_strategy)
        if (
            self.dispatch_class is None
            and config.combo_kernel_uniform_dispatch
            and self._detect_uniform_subkernels()
        ):
            self.dispatch_class = ComboKernel.UniformDispatch

        # Attempt uniform dispatch path
        if self.dispatch_class is ComboKernel.UniformDispatch:
            result = self._codegen_uniform_kernel(
                name,
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduction,
                size_hints_list,
            )
            if result is not None:
                return result
            # Verification failed, fall back to sequential dispatch.
            # Reset state that may have been modified during the uniform attempt.
            self.dispatch_class = ComboKernel.SequentialDispatch
            self.min_x_blocks_list = []
            self.x_numels_list = []
            self.y_tree_list = []
            self.block_args = []
            self.dynamic_shape_args = []
            self._uniform_dispatch_info = None
            log.debug(
                "ComboKernel: uniform dispatch verification failed, "
                "falling back to sequential dispatch"
            )

        code = IndentedBuffer()

        code.splice(self.triton_kernel_cls.gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        seen_helpers: OrderedSet[str] = OrderedSet()
        for sub_kernel in self.sub_kernels:
            for helper in sub_kernel.helper_functions:
                if helper not in seen_helpers:
                    code.writeline("")
                    code.splice(helper)
                    seen_helpers.add(helper)

        argdefs, _, signature, _ = self.args.python_argdefs()
        argdefs = self.add_numel_to_args(argdefs, signature)
        block_args = self.get_block_args()
        if not self.bake_blocks:
            argdefs.extend([ArgName(x.name, is_constexpr=True) for x in block_args])
            if triton_version_uses_attrs_dict():
                signature.extend(block_args)

        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=pointwise_with_reduction,
                signature=signature,
                argdefs=argdefs,
                size_hints_list=size_hints_list,
            )
        )
        kernel_name = name or str(Placeholder.KERNEL_NAME)
        code.writeline(
            f"def {kernel_name}({', '.join(x.full_name() for x in argdefs)}):"
        )

        with code.indent():
            if config.triton.proton_profiling:
                code.writeline(f'pl.enter_scope("{kernel_name}")')
            code.splice("pid = tl.program_id(0)")
            if self.bake_blocks:
                self.codegen_blocks(code)

            sub_kernel_codes = self._codegen_sub_kernel_bodies()
            shared_body = self._try_get_shared_body(
                sub_kernel_codes, signature, heuristics_list
            )
            if shared_body is not None:
                self._codegen_shared_branches(code, sub_kernel_codes, shared_body)
            else:
                for num, sub_kernel_code in enumerate(sub_kernel_codes):
                    self._codegen_branch(code, num, sub_kernel_code)

                code.splice("else:")
                with code.indent():
                    code.splice("pass")
            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=self._kernel_num_gb))

        return code.getvalue()

    def _codegen_uniform_kernel(
        self,
        name: str | None,
        heuristics: str,
        size_hints: dict[str, int],
        selected_kernel: TritonKernel,
        pointwise_with_reduction: bool,
        size_hints_list: list[dict[str, int]],
    ) -> str | None:
        """
        Generate the kernel code for uniform dispatch.
        Returns the kernel source string, or None if body verification fails.
        """
        # Step 1: Generate bodies for all sub-kernels
        for sub_kernel in self.sub_kernels:
            sub_kernel.codegen_body()
            sub_kernel._filter_pdl(sub_kernel.body)

        # Step 2: Verify bodies are truly identical and build slot mapping
        if not self._build_uniform_dispatch_info():
            return None

        assert self._uniform_dispatch_info is not None
        slots = self._uniform_dispatch_info["slots"]
        body_buf = self._uniform_dispatch_info["body"]

        # Step 3: Build modified argdefs/signature
        # Replace individual buffer args with slot pointer-table args
        orig_argdefs, _, orig_signature, _ = self.args.python_argdefs()

        # Collect all inner buffer names that are part of slots
        slot_inner_names: set[str] = set()
        for slot in slots:
            for refs in self._uniform_dispatch_info["buf_refs"]:
                for ref_name in refs:
                    slot_inner_names.add(ref_name)

        # Build new argdefs: slot pointer-table args + non-buffer args
        new_argdefs: list[ArgName] = []
        new_signature: list[Any] = []
        for slot_idx, slot in enumerate(slots):
            slot_arg_name = f"_slot_{slot_idx}_ptrs"
            new_argdefs.append(ArgName(slot_arg_name))
            # Signature entry for pointer arg (int64 tensor / pointer)
            new_signature.append(
                SizeArg(slot_arg_name, Integer(0))  # placeholder for signature_to_meta
            )

        # Add non-buffer args (size args, etc.)
        for argdef, sig_entry in zip(orig_argdefs, orig_signature):
            if argdef.name not in slot_inner_names:
                new_argdefs.append(argdef)
                new_signature.append(sig_entry)

        # Add numel args and block args
        new_argdefs = self.add_numel_to_args(new_argdefs, new_signature)
        block_args = self.get_block_args()
        if self.enable_autotune:
            new_argdefs.extend(
                [ArgName(x.name, is_constexpr=True) for x in block_args]
            )
            if triton_version_uses_attrs_dict():
                new_signature.extend(block_args)

        # Step 4: Generate the kernel code
        code = IndentedBuffer()
        code.splice(self.triton_kernel_cls.gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        seen_helpers: OrderedSet[str] = OrderedSet()
        for sub_kernel in self.sub_kernels:
            for helper in sub_kernel.helper_functions:
                if helper not in seen_helpers:
                    code.writeline("")
                    code.splice(helper)
                    seen_helpers.add(helper)

        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=pointwise_with_reduction,
                signature=new_signature,
                argdefs=new_argdefs,
                size_hints_list=size_hints_list,
            )
        )
        kernel_name = name or str(Placeholder.KERNEL_NAME)
        code.writeline(
            f"def {kernel_name}({', '.join(x.full_name() for x in new_argdefs)}):"
        )

        with code.indent():
            if config.triton.proton_profiling:
                code.writeline(f'pl.enter_scope("{kernel_name}")')
            code.splice("pid = tl.program_id(0)")
            if not self.enable_autotune:
                self.codegen_blocks(code)

            # Compute dispatch indices
            ref_sub = self.sub_kernels[0]
            # Get xnumel for computing blocks per kernel
            xnumel_expr: int | str = 0
            for tree in ref_sub.range_trees:
                if tree.prefix == "x":
                    simplified = V.graph.sizevars.simplify(tree.numel)
                    if isinstance(simplified, (Integer, int)):
                        xnumel_expr = int(simplified)
                    else:
                        xnumel_expr = f"xnumel_0"
                    break

            if ref_sub.no_x_dim:
                code.writeline(
                    f"num_blocks_per_kernel = {xnumel_expr}"
                )
            else:
                code.writeline(
                    f"num_blocks_per_kernel = tl.cdiv({xnumel_expr}, XBLOCK)"
                )
            code.writeline(
                "kernel_idx = pid // num_blocks_per_kernel"
            )
            code.writeline(
                "pid_offset = pid % num_blocks_per_kernel"
            )

            # Emit pointer loads from slot arrays
            for slot_idx, slot in enumerate(slots):
                inner_name = slot["inner_name"]
                dtype = slot["dtype"]
                tl_type = triton_type(dtype) if dtype is not None else "tl.int64"
                code.writeline(
                    f"{inner_name} = tl.load(_slot_{slot_idx}_ptrs + kernel_idx)"
                    f".to(tl.pointer_type({tl_type}))"
                )

            # Emit static numels for sub-kernel 0
            for tree in ref_sub.range_trees:
                simplified = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified, (Integer, int)):
                    code.writeline(f"{tree.prefix}numel = {int(simplified)}")

            # Emit the reduction block-size constexpr in the body ONLY for
            # persistent reductions. For a persistent reduction R0_BLOCK is a
            # baked compile-time constant (not a kernel parameter), so it must be
            # defined here. For a non-persistent (looped) reduction, R0_BLOCK is
            # instead a tl.constexpr KERNEL PARAMETER (added from get_block_args()
            # when autotuning, see above), so emitting it here as well would raise
            # "R0_BLOCK is already defined. constexpr cannot be reassigned." at
            # Triton compile time. This is the bug that broke uniform dispatch on
            # non-persistent / mixed reduction combo groups in training.
            for tree in ref_sub.range_trees:
                if tree.is_reduction and ref_sub.persistent_reduction:
                    simplified = V.graph.sizevars.simplify(tree.numel)
                    if isinstance(simplified, (Integer, int)):
                        rblock_size = next_power_of_2(int(simplified))
                        code.writeline(
                            f"R0_BLOCK: tl.constexpr = {rblock_size}"
                        )

            # Emit the single body (from sub-kernel 0)
            code.splice(body_buf)

            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=self._kernel_num_gb))

        return code.getvalue()

    def codegen_kernel_benchmark(self, num_gb: float) -> IndentedBuffer:
        """
        Generates Python code for benchmarking this combo kernel.
        - Creates example inputs (random tensors, constants, sizes).
        - Runs the kernel on the current GPU/stream.
        - Prints runtime (ms) and throughput (GB/s) using `num_gb`.
        Args:
            num_gb (float): The number of gigabytes to use for throughput calculation.
        Returns:
            IndentedBuffer: A buffer containing the generated Python benchmark code.
        """
        result = IndentedBuffer()
        _argdefs, call_args, signature, _ = self.args.python_argdefs()
        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.try_get_buffer(arg_name)
                if buf:
                    size = V.graph.sizevars.optimization_hints(buf.get_size())
                    stride = V.graph.sizevars.optimization_hints(buf.get_stride())
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{buf.get_device()}', dtype={buf.get_dtype()})"
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    size = V.graph.sizevars.optimization_hints(const_tensor.size())
                    stride = V.graph.sizevars.optimization_hints(const_tensor.stride())
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.optimization_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                elif isinstance(arg_sig, WorkspaceArg):
                    device = V.graph.get_current_device_or_throw()
                    count = V.graph.sizevars.optimization_hint(arg_sig.count)
                    # for benchmark harness, we ignore arg_sig.zero_mode and always zero it
                    result.writeline(
                        f"{var_name} = torch.zeros({count}, device='{device}', dtype={arg_sig.dtype})"
                    )
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            if self.dynamic_shape_args:
                var_names.extend(self.kernel_benchmark_extra_args())
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        device = V.graph.get_current_device_or_throw()
        index = V.graph.get_current_device_or_throw().index
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                stream_name = get_raw_stream_name(index)
                result.writeline(f"{stream_name} = get_raw_stream({index})")
                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                result.writeline(
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args)"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline(
                "from torch._inductor.runtime.benchmarking import benchmarker"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                f"ms = benchmarker.benchmark(call, fn_args=(args,), device='{device.type}',rep=40)"
            )
            result.writeline(f"num_gb = {num_gb}")
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def imports_for_benchmark_kernel(self) -> str:
        # Dedent BEFORE substituting get_raw_stream: a multi-line override would
        # otherwise collapse dedent's common prefix and misindent the imports.
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
            """
        ).format(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))

    def uniquify_block_sizes(
        self, code: IndentedBuffer, num_kernel: int, uniquify: list[str]
    ) -> IndentedBuffer:
        if not uniquify:
            return code
        modified = IndentedBuffer(initial_indent=code._indent)
        for line in code._lines:
            if isinstance(line, str) and (blocks := [e for e in uniquify if e in line]):
                modified_line = line
                for block in blocks:
                    modified_line = modified_line.replace(
                        block, f"{block}_{num_kernel}"
                    )
                modified.writeline(modified_line)
            elif isinstance(line, DeferredLineBase) and (
                blocks := [e for e in uniquify if e in line.line]
            ):
                modified_line = line.line
                for block in blocks:
                    modified_line = modified_line.replace(
                        block, f"{block}_{num_kernel}"
                    )
                new_line = line._new_line(modified_line)
                modified.writeline(new_line)
            else:
                modified.writeline(line)
        return modified

    def call_kernel(self, name: str) -> None:
        _, call_args, _, arg_types = self.args.python_argdefs()

        wrapper = V.graph.wrapper_code
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")

        if (
            self.dispatch_class is ComboKernel.UniformDispatch
            and self._uniform_dispatch_info is not None
        ):
            self._call_kernel_uniform(name, call_args, arg_types)
            return

        if self.dynamic_shape_args:
            self.add_numel_to_call_args(name, call_args, arg_types)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            triton_meta=self.triton_meta,
            inductor_meta=self.inductor_meta,
        )

    def _call_kernel_uniform(
        self, name: str, call_args: list[Any], arg_types: list[Any]
    ) -> None:
        """
        Emit the wrapper code for uniform dispatch: build pointer-table tensors
        and call the kernel with them.
        """
        assert self._uniform_dispatch_info is not None
        slots = self._uniform_dispatch_info["slots"]
        wrapper = V.graph.wrapper_code

        # Build the set of buffer call_args that are replaced by slots
        argdefs, _, _, _ = self.args.python_argdefs()
        # Map inner_name -> call_arg (outer buffer name)
        inner_to_call_arg: dict[str, str] = {}
        for argdef, call_arg in zip(argdefs, call_args):
            inner_to_call_arg[argdef.name] = call_arg

        # Determine device from first buffer in first slot
        first_buf = slots[0]["call_args"][0]
        device_expr = f"{first_buf}.device"

        # Materialize the per-slot GPU pointer tables. Delegated to a single
        # construction seam (extended for cudagraphs in the follow-up step). It
        # reuses a persistent pinned staging buffer + persistent device table per
        # call site, so no per-iteration cudaHostAlloc / device-alloc is paid --
        # this is what fixed the no_cudagraphs host overhead -- while staying
        # correct when one uniform kernel is reused across multiple sub-kernel
        # groups (see _emit_uniform_pointer_tables).
        slot_var_names = self._emit_uniform_pointer_tables(
            wrapper, slots, device_expr
        )

        # Build the new call_args: slot pointer tables + non-buffer args
        slot_inner_names: set[str] = set()
        for slot in slots:
            for refs in self._uniform_dispatch_info["buf_refs"]:
                for ref_name in refs:
                    slot_inner_names.add(ref_name)

        new_call_args: list[Any] = list(slot_var_names)
        new_arg_types: list[Any] = [None] * len(slot_var_names)  # tensor types

        # Add non-buffer args (size vars, workspace args, etc.)
        for argdef, call_arg, arg_type in zip(argdefs, call_args, arg_types):
            if argdef.name not in slot_inner_names:
                new_call_args.append(call_arg)
                new_arg_types.append(arg_type)

        # Add numel args if needed
        if self.dynamic_shape_args:
            self.add_numel_to_call_args(name, new_call_args, new_arg_types)

        wrapper.generate_kernel_call(
            name,
            new_call_args,
            triton=True,
            arg_types=new_arg_types,
            triton_meta=self.triton_meta,
            inductor_meta=self.inductor_meta,
        )

    def _emit_uniform_pointer_tables(
        self, wrapper: Any, slots: list[dict[str, Any]], device_expr: str
    ) -> list[str]:
        """
        Emit wrapper code that materializes the per-slot GPU pointer tables for
        one uniform-dispatch kernel call, and return the per-slot arg variable
        names to pass to the kernel.

        This is the single construction seam for the uniform pointer table.

        no_cudagraphs: the table *content* changes every iteration because
        intermediate buffers are re-allocated on each call, so it cannot be built
        once. Allocation is amortized instead: a persistent pinned staging buffer
        and a persistent device table are allocated ONCE per call site (keyed by a
        codegen-assigned id) and reused every iteration. Each call only fills the
        pinned buffer and issues a single async H2D copy, eliminating the
        per-iteration cudaHostAlloc + device alloc that dominated small many-group
        models (e.g. resnet18: 44 pin_memory()/iter -> 0 in steady state). A
        per-call-site CUDA event guards reuse so the host cannot overwrite the
        pinned buffer before its previous async copy has drained.

        cudagraphs: the wrapper runs only during warmup + capture (never on
        replay), and pool buffer addresses are stable across replays. The pinned
        staging buffer is persistent so the captured H2D copy has a valid, stable
        source on replay (a freed per-call pinned source was the original
        cudaErrorInvalidValue). The device table is allocated fresh so it is
        cudagraph-pool tracked during capture, and the event sync/record is
        skipped while the stream is capturing (illegal there). All of this lives
        in the once-emitted `_uniform_stage` helper; the regime is selected by the
        codegen-time `config.triton.cudagraphs` literal passed below.

        Distinct buffers per call site keep it correct when one uniform kernel is
        reused across multiple sub-kernel groups (each group is a separate call
        site with a distinct id).
        """
        self._emit_uniform_stage_helper_once(wrapper)

        # Every slot spans the same sub-kernels, so each slot occupies a
        # contiguous num_kernels-sized block in the flattened table.
        num_kernels = len(slots[0]["call_args"])
        flat_ptr_exprs: list[str] = []
        for slot in slots:
            for buf in slot["call_args"]:
                flat_ptr_exprs.append(f"{buf}.data_ptr()")

        # Unique id per uniform call site (a single uniform kernel may be called
        # for several groups; each group must get its own reusable buffers).
        call_site_id = getattr(wrapper, "_uniform_call_site_counter", 0)
        wrapper._uniform_call_site_counter = call_site_id + 1

        dev_var = f"_uniform_tbl_{call_site_id}"
        cudagraphs_literal = "True" if config.triton.cudagraphs else "False"
        wrapper.writeline(
            f"{dev_var} = _uniform_stage("
            f"{call_site_id}, [{', '.join(flat_ptr_exprs)}], {device_expr}, "
            f"{cudagraphs_literal})"
        )

        slot_var_names: list[str] = []
        for slot_idx in range(len(slots)):
            start = slot_idx * num_kernels
            end = start + num_kernels
            var_name = f"_uniform_slot_{slot_idx}_ptrs"
            wrapper.writeline(f"{var_name} = {dev_var}[{start}:{end}]")
            slot_var_names.append(var_name)
        return slot_var_names

    def _emit_uniform_stage_helper_once(self, wrapper: Any) -> None:
        """
        Emit the module-level uniform-dispatch staging pool + helper once per
        graph. The pool lives at wrapper module scope so it persists across
        call() invocations; each entry (per call site id) holds a reused pinned
        host staging buffer, a reused device pointer table, and a CUDA event that
        serializes host refills against the previous async H2D copy.
        """
        if getattr(wrapper, "_uniform_stage_helper_emitted", False):
            return
        wrapper._uniform_stage_helper_emitted = True
        wrapper.header.splice(
            """
_uniform_ptr_pool = {}


def _uniform_stage(_uid, _ptrs, _device, _cudagraphs):
    # Build the per-call-site GPU pointer table.
    #
    # The pinned host staging buffer is persistent (allocated once per call site,
    # never freed) so that a captured host->device copy stays valid on cudagraph
    # replay -- a per-call/freed pinned source was the original cudaErrorInvalidValue.
    #
    # Device table:
    #   * no_cudagraphs: persistent and reused (no per-iteration alloc); a CUDA
    #     event prevents the host from overwriting the pinned buffer before its
    #     previous async copy drains.
    #   * cudagraphs: allocated fresh so it is cudagraph-pool tracked when the
    #     wrapper runs during capture; the host wrapper does not run on replay, so
    #     the captured H2D re-copies the (stable) pinned addresses into the (stable)
    #     pool table each replay. Event sync/record is illegal during capture and
    #     is therefore skipped while capturing.
    _n = len(_ptrs)
    _ent = _uniform_ptr_pool.get(_uid)
    if _ent is None or _ent[0].numel() < _n:
        if _ent is not None:
            _ent[2].synchronize()
        _pinned = torch.empty(_n, dtype=torch.int64, device="cpu").pin_memory()
        _ent = [_pinned, None, torch.cuda.Event()]
        _uniform_ptr_pool[_uid] = _ent
    _pinned, _devtbl, _ev = _ent
    _capturing = torch.cuda.is_current_stream_capturing()
    if not _capturing:
        _ev.synchronize()
    _pinned[:_n].copy_(torch.tensor(_ptrs, dtype=torch.int64))
    if _cudagraphs or _devtbl is None or _devtbl.numel() < _n:
        _devtbl = torch.empty(_n, dtype=torch.int64, device=_device)
        if not _cudagraphs:
            _ent[1] = _devtbl
    _devtbl[:_n].copy_(_pinned[:_n], non_blocking=True)
    if not _capturing:
        _ev.record()
    return _devtbl
"""
        )

    def combo_grid_meta(self, size_hints_list: list[dict[str, int]]) -> dict[str, Any]:
        """
        Build metadata used by combo-kernel grid/dispatch/autotune helpers.
        """
        dynamic_shape = bool(self.dynamic_shape_args)
        num_kernels = len(self.sub_kernels)
        min_blocks = (
            max(self.min_x_blocks_list) * num_kernels if not dynamic_shape else None
        )

        meta: dict[str, Any] = {
            "num_kernels": num_kernels,
            "min_blocks": min_blocks,
            # Captured at codegen time so runtime sees the same value the
            # source was generated with, regardless of later config changes.
            "autotune_grouping": config.combo_kernel_autotune_grouping,
        }

        if self.bake_blocks or self.combo_compile_time_autotune:
            default_config: dict[str, int] = {}
            if self.combo_compile_time_autotune:
                # Compile-time autotune: per-subkernel winning block sizes are passed as args;
                # num_warps / num_stages / backend kwargs are autotuned over the distinct winner
                # launch candidates (flattened to tuples so the meta stays repr-serializable).
                if not self.combo_launch_candidates:
                    raise AssertionError(
                        "compile-time autotune requires at least one launch candidate"
                    )
                if self.stitched_block_config is not None:
                    default_config = dict(self.stitched_block_config)
                meta["stitched_launch_candidates"] = [
                    (c.kwargs, c.num_warps, c.num_stages)
                    for c in self.combo_launch_candidates
                ]
            elif self.no_bench_stitched_config is not None:
                stitched = self.no_bench_stitched_config
                default_config = {
                    k: int(v) for k, v in stitched.kwargs.items() if "BLOCK" in k
                }
                meta["stitched_backend_kwargs"] = {
                    k: v for k, v in stitched.kwargs.items() if "BLOCK" not in k
                }
                meta["stitched_num_warps"] = stitched.num_warps
                meta["stitched_num_stages"] = stitched.num_stages
            elif self.per_subkernel_blocks:
                # Per-subkernel block sizes: XBLOCK_0, XBLOCK_1, etc.
                for num, sub_kernel in enumerate(self.sub_kernels):
                    if sub_kernel.no_x_dim:
                        default_config[f"XBLOCK_{num}"] = 1
                    else:
                        block_size = (
                            self.block_size_2d
                            if any(self.y_tree_list)
                            else self.block_size_1d
                        )
                        default_config[f"XBLOCK_{num}"] = block_size

                    if self.y_tree_list[num]:
                        default_config[f"YBLOCK_{num}"] = self.block_size_2d
            else:
                if "YBLOCK" in self.block_args:
                    default_config = {
                        "XBLOCK": self.block_size_2d,
                        "YBLOCK": self.block_size_2d,
                    }
                else:
                    default_config = {"XBLOCK": self.block_size_1d}
            meta["default_config"] = default_config
        else:
            meta["default_config"] = None

        for num, sub_kernel in enumerate(self.sub_kernels):
            meta[f"no_x_dim_{num}"] = sub_kernel.no_x_dim

            if self.per_subkernel_blocks:
                meta[f"heuristic_{num}"] = (
                    "persistent_reduction"
                    if sub_kernel.persistent_reduction
                    else "reduction"
                    if sub_kernel.inside_reduction
                    else "pointwise"
                )

                meta[f"size_hints_{num}"] = size_hints_list[num]
                meta[f"inductor_meta_{num}"] = sub_kernel.inductor_meta_per_kernel()
                if meta[f"heuristic_{num}"] == "pointwise":
                    if len(size_hints_list[num]) == 2:
                        meta[f"tile_hint_{num}"] = "TileHint.SQUARE"
                    else:
                        meta[f"tile_hint_{num}"] = "TileHint.DEFAULT"
                else:
                    meta[f"reduction_hint_{num}"] = (
                        sub_kernel.features.get_reduction_hint(
                            sub_kernel.tiling_scores
                        ).name
                    )

            for tree in sub_kernel.range_trees:
                if not tree.is_reduction:
                    numel_name = f"{tree.prefix}numel_{num}"
                    if numel_name in self.dynamic_shape_args:
                        meta[numel_name] = None
                    else:
                        meta[numel_name] = int(V.graph.sizevars.simplify(tree.numel))

        return meta
