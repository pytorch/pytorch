import itertools
import logging
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, Optional, Union

import sympy
from sympy import Integer, Symbol

from torch.utils._ordered_set import OrderedSet

from .. import config, metrics
from ..runtime.hints import DeviceProperties
from ..runtime.runtime_utils import next_power_of_2
from ..runtime.triton_heuristics import (
    RoundRobinComboKernelGrid,
    SequentialComboKernelGrid,
)
from ..scheduler import BaseSchedulerNode
from ..utils import Placeholder, triton_version_uses_attrs_dict
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    DeferredLine,
    IndentedBuffer,
    InplacedBuffer,
    Kernel,
    PythonPrinter,
    RemovedArg,
    SizeArg,
    WorkspaceArg,
)
from .simd import prefix_is_reduction, SIMDScheduling
from .simd_kernel_features import SIMDKernelFeatures
from .triton import gen_common_triton_imports, TritonKernel
from .triton_utils import config_of, equal_1_arg_indices, signature_to_meta


log = logging.getLogger(__name__)
pexpr = PythonPrinter().doprint
LARGE_NUMELS = 512e5
BLOCK_UTILIZATION = 0.8


def _default_custom_combo_kernel_horizontal_partition(
    nodes: list[BaseSchedulerNode],
    triton_scheduling: SIMDScheduling,
    kernel_map: dict[BaseSchedulerNode, TritonKernel],
    node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
) -> list[list[BaseSchedulerNode]]:
    """Horizontally partition the given list of nodes into a list of list of nodes where each sublist
    represents a partition. Nodes in different partitions are implemented in different combo kernels.
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
    nodes_per_ndim: list[list[BaseSchedulerNode]] = []
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
        long_reduction = [
            n
            for n in reduction
            if (
                V.graph.sizevars.shape_env.has_hint(n.group[-1][-1])
                and V.graph.sizevars.size_hint(n.group[-1][-1]) > 2048  # type: ignore[arg-type]
            )
        ]
        short_reduction = [n for n in reduction if n not in long_reduction]
        if long_reduction:
            log.debug(
                "ComboKernels: %d long reduction nodes are separated",
                len(long_reduction),
            )
        large_pointwise = [
            n
            for n in not_reduction
            if not kernel_map[n].inside_reduction
            and len(kernel_map[n].numels) == 2
            and V.graph.sizevars.shape_env.has_hint(kernel_map[n].numels["x"])
            and V.graph.sizevars.size_hint(kernel_map[n].numels["x"]) > LARGE_NUMELS
        ]
        if large_pointwise:
            # TODO benchmark the performance when large pointwise nodes combining with others
            log.debug(
                "ComboKernels: %d large pointwise nodes are separated",
                len(large_pointwise),
            )
            not_reduction = [n for n in not_reduction if n not in large_pointwise]
            nodes_per_ndim.extend([node] for node in large_pointwise)

        nodes_per_ndim.extend(
            g for g in (not_reduction, short_reduction, long_reduction) if g
        )

    assert sum(len(p) for p in nodes_per_ndim) == len(nodes)
    return nodes_per_ndim


_custom_combo_kernel_horizontal_partition_algorithm: Callable[
    [
        list[BaseSchedulerNode],
        SIMDScheduling,
        dict[BaseSchedulerNode, TritonKernel],
        dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
    ],
    list[list[BaseSchedulerNode]],
] = _default_custom_combo_kernel_horizontal_partition


def set_custom_combo_kernel_horizontal_partition(
    algorithm: Callable[
        [
            list[BaseSchedulerNode],
            SIMDScheduling,
            dict[BaseSchedulerNode, TritonKernel],
            dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
        ],
        list[list[BaseSchedulerNode]],
    ],
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
    partitions: list[list[BaseSchedulerNode]]
    cur_partition: list[BaseSchedulerNode]
    cur_count: int

    def finalize(self) -> None:
        if self.cur_partition:
            self.partitions.append(self.cur_partition)


class ComboKernel(Kernel):
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
        node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
        custom_algorithm: bool,
    ) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        # TODO support combination of kernels with different block dimensions
        assert len(subkernel_nodes) >= 1
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_algorithm
        )

        ndim_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        yelem_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )

        for node in subkernel_nodes:
            _node_schedule, tiled_groups, _numel, _rnumel = node_info_map[node]
            node_info = node

            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)

            ndim = len(tiled_groups)
            assert ndim >= 2, f"Combokernel not support tile {tiled_groups}"
            if not mixed_sizes and ndim == 3:
                y_elem = tiled_groups["y"]
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
        nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        kernel_map: dict[BaseSchedulerNode, TritonKernel],
        node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
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

    def __init__(
        self, enable_autotune: bool = False, mixed_sizes: bool = False
    ) -> None:
        super().__init__()
        self.sub_kernels: list[TritonKernel] = []
        self.iter_vars_count = itertools.count()
        self.grids: list[list[int]] = []
        self.min_x_blocks_list: list[Union[int, str]] = []
        self.x_numels_list: list[Union[int, str]] = []
        self.enable_autotune = enable_autotune
        self.mixed_sizes = mixed_sizes
        self.dispatch_class: Optional[
            type[Union[ComboKernel.SequentialDispatch, ComboKernel.RoundRobinDispatch]]
        ] = None
        self.block_args: list[str] = []
        # there following are used when autotuning is disabled
        self.block_size_1d = 1024  # Try tuning this value
        self.block_size_2d = 32
        self.num_warps = 8
        self.block_size_reduce = 256
        self.dynamic_shape_args: list[str] = []

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
    ) -> TritonKernel:
        """
        Only allow optimize_mask=True when 1) sequential dispatch is used,
        2) numels except x dimension are the same for each sub kernel.
        """
        return TritonKernel(
            tiling,
            features=features,
            pid_cache={"tl.program_id(0)": "pid_offset"},
            optimize_mask=optimize_mask,
            is_combo_kernel=True,
            # foreach kernels don't work with cooperative reductions
            override_cooperative_reduction=False,
        )

    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> list[str]:
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
        uniquify_block_sizes = []
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if isinstance(simplified_tree_numel, (Integer, int)):
                code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")
            else:
                assert f"{tree.prefix}numel_{num}" in self.dynamic_shape_args
                uniquify_block_sizes.append(f"{tree.prefix}numel")

            if not tree.is_reduction:
                if isinstance(simplified_tree_numel, (Integer, int)):
                    grid.append(int(simplified_tree_numel))
                else:
                    # pyrefly: ignore [bad-argument-type]
                    grid.append(f"{tree.prefix}numel_{num}")

            if tree.is_reduction and sub_kernel.persistent_reduction:
                if isinstance(simplified_tree_numel, (Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    raise RuntimeError(
                        "Dynamic shape on reduction dimension is not supported"
                    )
                val = next_power_of_2(val)
                code.writeline(
                    f"{tree.prefix.upper()}BLOCK_{num}: tl.constexpr = {val}"
                )
                uniquify_block_sizes.append(f"{tree.prefix.upper()}BLOCK")

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
        min_x_blocks: Union[int, str] = 0
        x_numels: Union[int, str] = 0
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
            prefix: next_power_of_2(
                V.graph.sizevars.size_hint(
                    numel, fallback=config.unbacked_symint_fallback
                )
            )
            for prefix, numel in sub_kernel.numels.items()
            if not prefix_is_reduction(prefix) or sub_kernel.inside_reduction
        }
        if sub_kernel.persistent_reduction:
            assert sub_kernel.inside_reduction
            heuristics = "persistent_reduction"
        elif sub_kernel.inside_reduction:
            heuristics = "reduction"
        else:
            heuristics = "pointwise"
        return heuristics, size_hints

    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]:
        if not self.enable_autotune:
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
            assert num_reduction == 0, (
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
                    assert not isinstance(arg, RemovedArg)
                    mutated_args.add(arg)
        return sorted(mutated_args)

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is not None:
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
        pointwise_with_reduce: bool = False,
    ) -> str:
        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        triton_meta = {
            "signature": signature_to_meta(
                signature, size_dtype=size_dtype, argdefs=argdefs
            ),
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
        }
        triton_meta[
            "enable_fp_fusion"
        ] = not config.emulate_precision_casts  # pyrefly: ignore[unsupported-operation]

        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        # pyrefly: ignore [unsupported-operation]
        triton_meta["configs"] = [config_of(signature)]
        mutated_args = self.get_mutated_args_sub_kernels()
        dispatch = self.dispatch_class
        assert dispatch is not None
        inductor_meta = {
            "grid_type": dispatch.grid_expr.__name__,
            "combo_grid_meta": self.combo_grid_meta(),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            **TritonKernel.inductor_meta_common(),
        }

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
            reduction_hint = sub_kernel.features.get_reduction_hint()
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

        return heuristics_line

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        for block in self.block_args:
            assert block in (
                "XBLOCK",
                "YBLOCK",
                "R0_BLOCK",
            ), f"{block} is not supported without autotuning"
        if "YBLOCK" in self.block_args:
            code.splice(f"XBLOCK: tl.constexpr = {self.block_size_2d}")
            code.splice(f"YBLOCK: tl.constexpr = {self.block_size_2d}")
        else:
            code.splice(f"XBLOCK: tl.constexpr = {self.block_size_1d}")
        if "R0_BLOCK" in self.block_args:
            code.splice(f"R0_BLOCK: tl.constexpr = {self.block_size_reduce}")
            code.splice(f"RBLOCK: tl.constexpr = {self.block_size_reduce}")

    def get_block_args(self) -> list[ConstexprArg]:
        """
        Calculate blocks from sub_kernels and range_trees.
        **Update self.block_args**
        Return the block args
        """
        block_names = {}
        for sub_kernel in self.sub_kernels:
            # TODO: we assume all sub_kernels have the same block size
            for tree in sub_kernel.range_trees:
                if tree.is_reduction and (
                    not sub_kernel.inside_reduction or sub_kernel.persistent_reduction
                ):
                    continue
                if tree.prefix == "x" and sub_kernel.no_x_dim:
                    continue
                block_names[f"{tree.prefix.upper()}BLOCK"] = tree.prefix
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
                        str(
                            V.graph.sizevars.size_hint(
                                tree.numel, fallback=config.unbacked_symint_fallback
                            )
                        )
                    )
        return extra_args

    def codegen_kernel(self, name: Optional[str] = None) -> str:
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
        code = IndentedBuffer()

        code.splice(gen_common_triton_imports())
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
        if self.enable_autotune:
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
            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=0))

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
                    size = V.graph.sizevars.size_hints(
                        buf.get_size(), fallback=config.unbacked_symint_fallback
                    )
                    stride = V.graph.sizevars.size_hints(
                        buf.get_stride(), fallback=config.unbacked_symint_fallback
                    )
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    size = V.graph.sizevars.size_hints(
                        const_tensor.size(), fallback=config.unbacked_symint_fallback
                    )
                    stride = V.graph.sizevars.size_hints(
                        const_tensor.stride(), fallback=config.unbacked_symint_fallback
                    )
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                elif isinstance(arg_sig, WorkspaceArg):
                    device = V.graph.get_current_device_or_throw()
                    count = V.graph.sizevars.size_hint(arg_sig.count)
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
                stream_name = f"stream{index}"
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
                f"ms = benchmarker.benchmark(call, fn_args=(args,), device={device.type},rep=40)"
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
        """.format(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))
        )

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
        if self.dynamic_shape_args:
            self.add_numel_to_call_args(name, call_args, arg_types)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
        )

    def combo_grid_meta(self) -> dict[str, Any]:
        dynamic_shape = bool(self.dynamic_shape_args)
        num_kernels = len(self.sub_kernels)
        min_blocks = (
            max(self.min_x_blocks_list) * num_kernels if not dynamic_shape else None
        )

        if not self.enable_autotune:
            if "YBLOCK" in self.block_args:
                default_config = {
                    "XBLOCK": self.block_size_2d,
                    "YBLOCK": self.block_size_2d,
                }
            else:
                default_config = {"XBLOCK": self.block_size_1d}
        else:
            default_config = None

        meta = {
            "num_kernels": num_kernels,
            "min_blocks": min_blocks,
            "default_config": default_config,
        }

        for num, sub_kernel in enumerate(self.sub_kernels):
            meta[f"no_x_dim_{num}"] = sub_kernel.no_x_dim
            for tree in sub_kernel.range_trees:
                if not tree.is_reduction:
                    numel_name = f"{tree.prefix}numel_{num}"
                    if numel_name in self.dynamic_shape_args:
                        meta[numel_name] = None
                    else:
                        meta[numel_name] = int(V.graph.sizevars.simplify(tree.numel))

        return meta
