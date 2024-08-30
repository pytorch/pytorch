import itertools
import logging
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

from sympy import Integer

from torch.utils._ordered_set import OrderedSet

from .. import config, metrics
from ..runtime.hints import DeviceProperties, ReductionHint
from ..runtime.runtime_utils import next_power_of_2
from ..runtime.triton_heuristics import grid_combo_kernels
from ..scheduler import BaseSchedulerNode
from ..utils import Placeholder
from ..virtualized import V
from .common import DeferredLine, IndentedBuffer, Kernel, PythonPrinter, SizeArg
from .simd import SIMDScheduling
from .triton import gen_common_triton_imports, TritonKernel
from .triton_utils import config_of, signature_to_meta


log = logging.getLogger(__name__)
pexpr = PythonPrinter().doprint
LARGE_NUMELS = 512e5
BLOCK_UTILIZATION = 0.8


def _default_custom_combo_kernel_horizontal_partition(
    nodes: List[BaseSchedulerNode],
    triton_scheduling: SIMDScheduling,
    kernel_map: Dict[BaseSchedulerNode, TritonKernel],
    node_info_map: Dict[BaseSchedulerNode, Tuple[Any, Any, Any, Any]],
) -> List[List[BaseSchedulerNode]]:
    """Horizontally partition the given list of nodes into a list of list of nodes where each sublist
    represents a partion. Nodes in different partitions are implemented in different combo kernels.
    Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    Input arguments:
        nodes: a list of fused scheduler nodes to partition.
        triton_scheduling: TritonScheduling instance.
        kernel_map: a map from node to its kernel.
        node_info_map: a map from node to (node_schedule, tiled_groups, numel, rnumel).
    Output:
        a list of list of nodes with each sublist representing a partition.

    The default algorithm is to partition nodes based on the following rules:
        1) nodes with the same number of block dimensions are grouped together.
        2) large pointwise nodes (numels greater than LARGE_NUMELS) are separated from other nodes.
        3) large reduce nodes are separated from other nodes.
    """

    assert len(nodes) >= 1

    # first partition nodes based on number of block dimensions
    tilings = [node_info_map[n][1] for n in nodes]

    max_dims = max(len(t) for t in tilings)
    nodes_per_ndim = []
    for i in range(2, max_dims + 1):
        group_per_dim = [n for n, t in zip(nodes, tilings) if len(t) == i]
        reduction = [
            n
            for n in group_per_dim
            if kernel_map[n].inside_reduction
            and not (kernel_map[n].persistent_reduction and kernel_map[n].no_x_dim)
        ]
        not_reduction = [n for n in group_per_dim if n not in reduction]
        # rnumel > 2048 usually has long execution time
        # BaseSchedulerNode.group[-1][-1] is rnumel for reduction nodes
        long_reduction = [n for n in reduction if cast(Integer, n.group[-1][-1]) > 2048]
        short_reduction = [n for n in reduction if n not in long_reduction]
        if long_reduction:
            log.warning(
                "ComboKernels: %d long reduction nodes are separated",
                len(long_reduction),
            )
        large_pointwise = [
            n
            for n in not_reduction
            if not kernel_map[n].inside_reduction
            and len(kernel_map[n].numels) == 2
            and V.graph.sizevars.size_hint(kernel_map[n].numels[0]) > LARGE_NUMELS
        ]
        if large_pointwise:
            # TODO benchmark the performance when large pointwise nodes combining with others
            log.warning(
                "ComboKernels: %d large pointwise nodes are separated",
                len(large_pointwise),
            )
            not_reduction = [n for n in not_reduction if n not in large_pointwise]
            for node in large_pointwise:
                nodes_per_ndim.append([node])

        for g in (not_reduction, short_reduction, long_reduction):
            if g:
                nodes_per_ndim.append(g)

    assert sum(len(p) for p in nodes_per_ndim) == len(nodes)
    return nodes_per_ndim


_custom_combo_kernel_horizontal_partition_algorithm: Callable[
    [
        List[BaseSchedulerNode],
        SIMDScheduling,
        Dict[BaseSchedulerNode, TritonKernel],
        Dict[BaseSchedulerNode, Tuple[Any, Any, Any, Any]],
    ],
    List[List[BaseSchedulerNode]],
] = _default_custom_combo_kernel_horizontal_partition


def set_custom_combo_kernel_horizontal_partition(
    algorithm: Callable[
        [
            List[BaseSchedulerNode],
            SIMDScheduling,
            Dict[BaseSchedulerNode, TritonKernel],
            Dict[BaseSchedulerNode, Tuple[Any, Any, Any, Any]],
        ],
        List[List[BaseSchedulerNode]],
    ]
) -> None:
    """Sets the algorithm used to partition nodes into horizontal partitions. Nodes in different partitions
    are implemented in different combo kernels. Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restricts like CUDA limits for number of args.

    The algorithm should take a list of nodes and return a list of list of nodes.

    The default algorithm is to partition nodes based on number of block dimensions.
    """
    global _custom_combo_kernel_horizontal_partition_algorithm
    _custom_combo_kernel_horizontal_partition_algorithm = algorithm


@dataclass
class PartitionState:
    partitions: List[List[BaseSchedulerNode]]
    cur_partition: List[BaseSchedulerNode]
    cur_count: int

    def finalize(self) -> None:
        if self.cur_partition:
            self.partitions.append(self.cur_partition)


class ComboKernel(Kernel):
    MAX_NUM_ARGS = 250  # number where I would no longer get triton errors

    @staticmethod
    def _update_partition(
        partition_state: PartitionState,
        node_rw_count: int,
        node_info: BaseSchedulerNode,
    ) -> None:
        if partition_state.cur_count + node_rw_count > ComboKernel.MAX_NUM_ARGS:
            partition_state.partitions.append(partition_state.cur_partition)
            partition_state.cur_partition = [node_info]
            partition_state.cur_count = node_rw_count
        else:
            partition_state.cur_count += node_rw_count
            partition_state.cur_partition.append(node_info)

    @staticmethod
    def _base_horizontal_partition(
        subkernel_nodes: List[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        node_info_map: Dict[BaseSchedulerNode, Tuple[Any, Any, Any, Any]],
        custom_algorithm: bool,
    ) -> List[List[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        # TODO support combination of kernels with different block dimensions
        assert len(subkernel_nodes) >= 1
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_algorithm
        )

        ndim_to_partition_state: Dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        yelem_to_partition_state: Dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )

        for node in subkernel_nodes:
            node_schedule, tiled_groups, numel, rnumel = node_info_map[node]
            node_info = node

            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)

            ndim = len(tiled_groups)
            assert ndim >= 2, f"Combokernel not support tile {tiled_groups}"
            if not mixed_sizes and ndim == 3:
                y_elem = tiled_groups[0]
                partition_state = yelem_to_partition_state[y_elem]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )
            else:
                assert mixed_sizes or ndim <= 3, f"No mixed sizes: tile {tiled_groups}"
                partition_state = ndim_to_partition_state[ndim]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )

        all_partitions = []
        for partition_state in ndim_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)
        for partition_state in yelem_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)

        return all_partitions

    @staticmethod
    def horizontal_partition(
        nodes: List[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        kernel_map: Dict[BaseSchedulerNode, TritonKernel],
        node_info_map: Dict[BaseSchedulerNode, Tuple[Any, Any, Any, Any]],
        custom_algorithm: bool = False,
    ) -> List[List[BaseSchedulerNode]]:
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
                nodes, triton_scheduling, kernel_map, node_info_map
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
                    code.splice(f"pid_offset = pid - num_xblocks_{num-1}")

        @classmethod
        def _calculate_xblocks(
            cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            x_numels_list = kernel.x_numels_list
            for i in range(len(x_numels_list)):
                xnumels = (
                    x_numels_list[i]
                    if x_numels_list[i] > 0
                    else kernel.min_x_blocks_list[i]
                )
                xblock_str = (
                    f"tl.cdiv({xnumels}, XBLOCK)"
                    if x_numels_list[i] > 0
                    else f"{xnumels}"
                )
                if i == 0:
                    code.splice(f"num_xblocks_{i} = {xblock_str}")
                else:
                    code.splice(f"num_xblocks_{i} = num_xblocks_{i-1} + {xblock_str}")

        @classmethod
        def grid(
            cls, sub_kernel_numels: List[List[int]], x_blocks_list: List[int]
        ) -> Tuple[Any, ...]:
            xnumel = x_blocks_list
            ynumel = [e[-2] if len(e) > 1 else None for e in sub_kernel_numels]
            znumel = [e[-3] if len(e) > 2 else None for e in sub_kernel_numels]

            # TODO: improve 1d/2d mixed cases
            ynumel = (
                None if any(e is None for e in ynumel) else max(cast(List[int], ynumel))
            )
            znumel = (
                None if any(e is None for e in znumel) else max(cast(List[int], znumel))
            )

            numels = (
                (xnumel,)
                if not ynumel
                else (ynumel, xnumel)
                if not znumel
                else (znumel, ynumel, xnumel)
            )
            return numels

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

        @classmethod
        def grid(
            cls, sub_kernel_numels: List[List[int]], x_blocks_list: List[int]
        ) -> Tuple[Any, ...]:
            xnumel = [e[-1] if len(e) > 0 else None for e in sub_kernel_numels]
            ynumel = [e[-2] if len(e) > 1 else None for e in sub_kernel_numels]
            znumel = [e[-3] if len(e) > 2 else None for e in sub_kernel_numels]

            # TODO: support 1d/2d mixed cases
            xnumel = (
                None if any(e is None for e in xnumel) else max(cast(List[int], xnumel))
            )
            ynumel = (
                None if any(e is None for e in ynumel) else max(cast(List[int], ynumel))
            )
            znumel = (
                None if any(e is None for e in znumel) else max(cast(List[int], znumel))
            )

            numels = (
                (xnumel,)
                if not ynumel
                else (ynumel, xnumel)
                if not znumel
                else (znumel, ynumel, xnumel)
            )
            return numels

    def __init__(
        self, enable_autotune: bool = False, mixed_sizes: bool = False
    ) -> None:
        super().__init__()
        self.sub_kernels: List[TritonKernel] = []
        self.iter_vars_count = itertools.count()
        self.grids: List[List[int]] = []
        self.min_x_blocks_list: List[int] = []
        self.x_numels_list: List[int] = []
        self.enable_autotune = enable_autotune
        self.mixed_sizes = mixed_sizes
        self.dispatch_class: Optional[
            Union[
                Type[ComboKernel.SequentialDispatch],
                Type[ComboKernel.RoundRobinDispatch],
            ]
        ] = None
        self.block_args: List[str] = []
        # there following are used when autotuning is disabled
        self.block_size_1d = 1024  # Try tuning this value
        self.block_size_2d = 32
        self.num_warps = 8
        self.block_size_reduce = 256

    def create_sub_kernel(self, triton_kernel: TritonKernel) -> TritonKernel:
        sub_kernel = triton_kernel
        metrics.generated_kernel_count -= 1
        sub_kernel.args = self.args
        sub_kernel.iter_vars_count = self.iter_vars_count
        sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        self.sub_kernels.append(sub_kernel)
        return sub_kernel

    @staticmethod
    def create_triton_kernel(
        *groups: Any,
        index_dtype: str,
        mutations: OrderedSet[str],
        reduction_hint: ReductionHint,
        optimize_mask: bool,
    ) -> TritonKernel:
        return TritonKernel(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache={"tl.program_id(0)": "pid_offset"},
            reduction_hint=reduction_hint,
            optimize_mask=optimize_mask,
        )

    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> List[str]:
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        grid = []
        uniquify_block_sizes = []
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            if tree.prefix != "r":
                grid.append(int(simplified_tree_numel))

            if tree.prefix == "r" and sub_kernel.persistent_reduction:
                if isinstance(simplified_tree_numel, (Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                val = next_power_of_2(val)
                code.writeline(f"RBLOCK_{num}: tl.constexpr = {val}")
                uniquify_block_sizes.append("RBLOCK")

            if tree.prefix == "x" and sub_kernel.no_x_dim:
                code.writeline(f"XBLOCK_{num}: tl.constexpr = 1")
                uniquify_block_sizes.append("XBLOCK")
        self.grids.append(grid)
        return uniquify_block_sizes

    def min_x_blocks_sub_kernel(self, sub_kernel: TritonKernel, num: int) -> None:
        """
        Kernels with no_x_dim being true has no tunable XBLOCK. They have a fixed number of X blocks.
        Grid calculation needs to make sure that they are assigned with enough number of blocks.
        """
        min_x_blocks = 0
        x_numels = 0
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if tree.prefix == "x":
                if sub_kernel.no_x_dim:
                    min_x_blocks = int(simplified_tree_numel)
                    x_numels = -min_x_blocks
                else:
                    x_numels = int(simplified_tree_numel)
        self.min_x_blocks_list.append(min_x_blocks)
        self.x_numels_list.append(x_numels)

    def select_heuristics(self, sub_kernel: TritonKernel) -> Tuple[str, List[int]]:
        size_hints = [
            next_power_of_2(V.graph.sizevars.size_hint(numel))
            for numel in sub_kernel.numels
        ]
        if sub_kernel.persistent_reduction:
            assert sub_kernel.inside_reduction
            heuristics = "persistent_reduction"
        elif sub_kernel.inside_reduction:
            heuristics = "reduction"
        else:
            size_hints.pop()
            heuristics = "pointwise"
        return heuristics, size_hints

    def select_combo_heuristics(
        self, heuristics_list: List[str], size_hints_list: List[List[int]]
    ) -> Tuple[str, List[int], TritonKernel]:
        if not self.enable_autotune:
            return "foreach", size_hints_list[0], self.sub_kernels[0]
        if "reduction" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1][0] if heuristics_list[x[0]] == "reduction" else 0,
            )
            return heuristics_list[i], size_hints_list[i], self.sub_kernels[i]
        elif "pointwise" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1][0] if heuristics_list[x[0]] == "pointwise" else 0,
            )
            # modify size_hint to avoid oom check fail (may be a false alarm)
            num_pointwise = len([e for e in heuristics_list if e == "pointwise"])
            num_reduction = len([e for e in heuristics_list if e == "reduction"])
            num_persistent_reduction = len(
                [e for e in heuristics_list if e == "persistent_reduction"]
            )
            assert (
                num_reduction == 0
            ), "combining pointwise and reduction are not supported yet."
            heuristics = (
                "pointwise_with_reduction"
                if num_persistent_reduction > 0
                else "pointwise"
            )
            if len(heuristics_list) - num_pointwise >= 4:
                size_hints = size_hints_list[i]
                size_hints[0] = min(128, size_hints[0])
            return heuristics, size_hints_list[i], self.sub_kernels[i]
        else:
            return heuristics_list[0], size_hints_list[0], self.sub_kernels[0]

    def get_mutated_args_sub_kernels(self) -> List[str]:
        mutated_args = set()
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
                        sub_kernel.args.inplace_buffers[mutation].inner_name
                    )
                if mutation in sub_kernel.args.output_buffers:
                    mutated_args.add(sub_kernel.args.output_buffers[mutation])
        return sorted(mutated_args)

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is not None:
            return
        if not self.mixed_sizes:
            self.dispatch_class = ComboKernel.SequentialDispatch
            return
        # A negative x_blocks_list element means the kernel is not tunable,
        # i.e., no_x_dim = True
        x_numels_list = [abs(e) for e in self.x_numels_list]
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
        size_hints: List[int],
        selected_kernel: TritonKernel,
        pointwise_with_reduce: bool = False,
    ) -> str:
        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        _, _, signature, _ = self.args.python_argdefs()
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=size_dtype),
            "device": DeviceProperties.create(
                V.graph.scheduler.get_current_device_or_throw()
            ),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        mutated_args = self.get_mutated_args_sub_kernels()
        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            **TritonKernel.inductor_meta_common(),
        }

        sub_kernel = selected_kernel
        if heuristics == "foreach":
            heuristics_line = f"""
                @triton_heuristics.foreach(
                    num_warps={self.num_warps},
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                )
                @triton.jit
            """
        elif sub_kernel.inside_reduction:
            reduction_hint = sub_kernel.reduction_hint
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta}
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

        return heuristics_line

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        for block in self.block_args:
            assert block in [
                "XBLOCK",
                "YBLOCK",
                "RBLOCK",
            ], f"{block} is not supported without autotuning"
        if "YBLOCK" in self.block_args:
            code.splice(f"XBLOCK: tl.constexpr = {self.block_size_2d}")
            code.splice(f"YBLOCK: tl.constexpr = {self.block_size_2d}")
        else:
            code.splice(f"XBLOCK: tl.constexpr = {self.block_size_1d}")
        if "RBLOCK" in self.block_args:
            code.splice(f"RBLOCK: tl.constexpr = {self.block_size_reduce}")

    def add_blockd_to_args(self, argdefs: List[str]) -> List[str]:
        block_args = {}
        block_names = {}
        for num, sub_kernel in enumerate(self.sub_kernels):
            # TODO: we assume all sub_kernels have the same block size
            for tree in sub_kernel.range_trees:
                if tree.prefix == "r" and (
                    not sub_kernel.inside_reduction or sub_kernel.persistent_reduction
                ):
                    continue
                if tree.prefix == "x" and sub_kernel.no_x_dim:
                    continue
                block_args[f"{tree.prefix.upper()}BLOCK : tl.constexpr"] = tree.prefix
                block_names[f"{tree.prefix.upper()}BLOCK"] = tree.prefix
        if self.enable_autotune:
            argdefs.extend(block_args)
        self.block_args = list(block_names.keys())
        return argdefs

    def codegen_kernel(self, name: Optional[str] = None) -> str:
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
        code = IndentedBuffer()

        code.splice(gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, _, _ = self.args.python_argdefs()
        argdefs = self.add_blockd_to_args(argdefs)
        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=pointwise_with_reduction,
            )
        )
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )

        with code.indent():
            code.splice("pid = tl.program_id(0)")
            if not self.enable_autotune:
                self.codegen_blocks(code)

            for num, sub_kernel in enumerate(self.sub_kernels):
                assert self.dispatch_class is not None
                self.dispatch_class.codegen_pid_range(self, num, code)
                with code.indent():
                    uniquify = self.codegen_static_numels_sub_kernel(
                        code, sub_kernel, num
                    )
                    sub_kernel.codegen_body()
                    uniquified_body = self.uniquify_block_sizes(
                        sub_kernel.body, num, uniquify
                    )
                    code.splice(uniquified_body)

            code.splice("else:")
            with code.indent():
                code.splice("pass")

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=0))

        return code.getvalue()

    def codegen_kernel_benchmark(
        self, num_gb: float, grid: Optional[List[Any]] = None
    ) -> IndentedBuffer:
        result = IndentedBuffer()
        argdefs, call_args, signature, _ = self.args.python_argdefs()

        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        if grid is None:
            assert self.dispatch_class is not None
            grid_tuple = self.dispatch_class.grid(self.grids, self.x_numels_list)
            grid_str = ", ".join(pexpr(item) for item in grid_tuple)
            grid_extra_kwargs = (
                f"num_kernels={len(self.sub_kernels)}, "
                f"min_blocks={max(self.min_x_blocks_list) * len(self.sub_kernels)}, "
                f"is_sequential={self.dispatch_class is self.SequentialDispatch}"
            )
            grid_str = f"{grid_str}, {grid_extra_kwargs}"
            grid_arg = f"grid=grid_combo_kernels({grid_str})"
        else:
            grid_arg = f"grid={grid}"
        index = V.graph.scheduler.get_current_device_or_throw().index
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                stream_name = f"stream{index}"
                result.writeline(f"{stream_name} = get_raw_stream({index})")
                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, {grid_arg}, stream={stream_name})"
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
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {grid_arg})"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline(
                "from torch._inductor.runtime.benchmarking import benchmarker"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)"
            )
            result.writeline(f"num_gb = {num_gb}")
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def imports_for_benchmark_kernel(self) -> str:
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
            from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def uniquify_block_sizes(
        self, code: IndentedBuffer, num_kernel: int, uniquify: List[str]
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
            elif isinstance(line, DeferredLine) and (
                blocks := [e for e in uniquify if e in line.line]
            ):
                modified_line = line.line
                for block in blocks:
                    modified_line = modified_line.replace(
                        block, f"{block}_{num_kernel}"
                    )
                new_line = DeferredLine(line.name, modified_line)
                modified.writeline(new_line)
            else:
                modified.writeline(line)
        return modified

    def call_kernel(self, code: IndentedBuffer, name: str) -> None:
        _, call_args, _, arg_types = self.args.python_argdefs()

        wrapper = V.graph.wrapper_code
        assert self.dispatch_class is not None
        grid = self.dispatch_class.grid(self.grids, self.x_numels_list)
        num_kernels = len(self.sub_kernels)
        min_blocks = max(self.min_x_blocks_list) * num_kernels
        is_sequential = self.dispatch_class is self.SequentialDispatch
        if not self.enable_autotune:
            launch_grid = self.grid_no_autotune(
                grid, num_kernels, min_blocks, is_sequential
            )
            V.graph.wrapper_code.generate_kernel_call(
                name,
                call_args,
                grid=launch_grid,
                arg_types=arg_types,
                grid_fn="",
            )
            return
        # autotuning is enabled
        grid = wrapper.generate_default_grid(
            name,
            list(grid),
            grid_callable=grid_combo_kernels,
            num_kernels=num_kernels,
            min_blocks=min_blocks,
            is_sequential=is_sequential,
        )
        wrapper.generate_kernel_call(
            name,
            call_args,
            grid,
            V.graph.scheduler.get_current_device_or_throw().index,
            cuda=True,
            triton=True,
            arg_types=arg_types,
            grid_fn="grid_combo_kernels",
            grid_extra_kwargs=(
                f"num_kernels={num_kernels}, "
                f"min_blocks={min_blocks}, "
                f"is_sequential={is_sequential}"
            ),
        )

    def grid_no_autotune(
        self,
        grid: Tuple[Any],
        num_kernels: int,
        min_blocks: int,
        is_sequential: bool,
    ) -> List[int]:
        if "YBLOCK" in self.block_args:
            meta = {"XBLOCK": self.block_size_2d, "YBLOCK": self.block_size_2d}
        else:
            meta = {"XBLOCK": self.block_size_1d}
        grid_func = grid_combo_kernels(
            *grid,
            num_kernels=num_kernels,
            min_blocks=min_blocks,
            is_sequential=is_sequential,
        )
        return grid_func(meta)
