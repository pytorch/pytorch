# mypy: allow-untyped-defs
import functools

import sympy

from torch._inductor import config
from torch._inductor.codegen.simd import IterationRangesRoot, prefix_is_reduction
from torch._inductor.codegen.triton import (
    triton_compute_type,
    TritonCSEVariable,
    TritonKernel,
)
from torch._inductor.runtime.triton_heuristics import SplitScanGrid
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv

from ..utils import sympy_product, upcast_compute_type


class TritonSplitScanKernel(TritonKernel):
    """Generates a triton kernel that supports ops.scan calls while also splitting
    the reduction dimension over multiple triton programs.

    For this kernel, loop numels will always take the form ``(xdim, rdim)``
    and the grid has the shape ``(CeilDiv(rdim, RBLOCK), xdim)``. Communication
    between blocks occurs within a global memory workspace buffer, which
    must be zero-filled before launching the kernel.

    Note that generation for ``ops.reduction`` is not supported.

    For details of the communication strategy, see
    https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    """

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        pid_cache=None,
        fixed_config=None,
        **kwargs,
    ) -> None:
        if pid_cache is not None:
            raise AssertionError("not supported")
        if fixed_config is not None:
            raise AssertionError("not supported")
        super().__init__(
            tiling,
            **kwargs,
        )
        self.no_x_dim = True

    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False

    def initialize_range_tree(self, pid_cache):
        prefixes = ["y", "x", "r0_"]
        if len(self.numels) > len(prefixes):
            raise AssertionError("z dimension not supported for split scan")
        active_prefixes = prefixes[len(prefixes) - len(self.numels) :]

        grid_dims = {"r0_": 0, "x": 1, "y": 2}
        for prefix in active_prefixes:
            numel = self.numels[prefix]
            tensor_dim = 0 if prefix_is_reduction(prefix) else None
            grid_dim = grid_dims[prefix]
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    numel,
                    prefix,
                    grid_dim,
                    self,  # type: ignore[arg-type]
                    pid_cache=pid_cache,
                    is_loop=False,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim=False,
                )
            )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError("NYI TritonSplitDimKernel reductions")

    def scan(self, dtypes, combine_fn, values):
        """
        Perform an associative scan on 'values'.

        Supports one or two carried states.
        *) single lane: value+flag in one <=32-bit word, or 3 u64 slots for 64-bit)
        *) two lanes: 5-u64-slot layout ([flag, bv0, bv1, ip0, ip1])
        """
        import triton.language as tl

        num_lanes = len(values)
        if num_lanes not in (1, 2):
            raise AssertionError(
                f"TritonSplitScanKernel supports 1 or 2 lanes, got {num_lanes}"
            )

        dtypes = tuple(upcast_compute_type(dtype) for dtype in dtypes)
        compute_types = [triton_compute_type(dtype) for dtype in dtypes]
        element_nbits = [getattr(tl, ct[3:]).primitive_bitwidth for ct in compute_types]

        if num_lanes == 1:
            nbits = element_nbits[0]
            scratch_type = "tl.uint32" if nbits <= 16 else "tl.uint64"
            scratch_elems_per_block = 3 if nbits == 64 else 1
        else:
            scratch_type = "tl.uint64"
            scratch_elems_per_block = 5
        scratch_type_triton = getattr(tl, scratch_type[3:])
        scratch_nbytes_per_block = scratch_elems_per_block * (
            scratch_type_triton.primitive_bitwidth // 8
        )

        cse_compute = functools.partial(self.cse.generate, self.compute)

        if len(self.numels) != 2:
            raise AssertionError("Unexpected tiling")
        min_rblock = config.triton.min_split_scan_rblock
        reduction_numel = sympy_product(
            numel
            for prefix, numel in self.numels.items()
            if prefix_is_reduction(prefix)
        )
        pointwise_numel = sympy_product(
            numel
            for prefix, numel in self.numels.items()
            if not prefix_is_reduction(prefix)
        )
        max_blocks = pointwise_numel * CeilDiv(reduction_numel, min_rblock)
        nbytes = scratch_nbytes_per_block * max_blocks
        scratch_base: str | TritonCSEVariable
        scratch_base, _, offset = self.args.workspace(nelem=nbytes, zero_fill=True)
        if offset != 0:
            scratch_base = f"({scratch_base} + {self.index_to_str(offset)})"
        runtime_rblocks = f"tl.num_programs({self.range_trees[-1].index})"
        scratch_base = (
            f"{scratch_base}.to(tl.pointer_type({scratch_type})) + xoffset * "
            f"{scratch_elems_per_block} * {runtime_rblocks}"
        )

        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        if self._load_mask:
            raise AssertionError("ops.scan not supported inside ops.masked")

        broadcast_values = []
        for value, dtype, compute_type in zip(values, dtypes, compute_types):
            value = cse_compute(
                f"{value}.to({compute_type})",
                dtype=dtype,
                shape=value.shape,
            )
            value = cse_compute(
                f"tl.broadcast_to({value}, {self.dense_size_str()})",
                dtype=dtype,
                shape=self.dense_size_list(),
            )
            broadcast_values.append(value)

        combine_helper_fn = self._lift_helper(
            combine_fn, tuple(broadcast_values), dtypes
        )
        dim = self.triton_tensor_ndim() - 1
        if dim != 0:
            raise AssertionError(f"expected scan dim == 0, got {dim}")
        scan_shape = broadcast_values[0].shape
        if scan_shape is None:
            raise AssertionError("expected value.shape to be set")
        reduced_shape = list(scan_shape)
        del reduced_shape[dim]

        def newvars(shape):
            return [self.cse.newvar(dtype=dtype, shape=shape) for dtype in dtypes]

        def csv(vars_):
            # trailing-comma form so a single lane still reads as a 1-tuple
            return "".join(f"{v}, " for v in vars_)

        # block_scan is the in-order inclusive scan of this block; its last
        # element is the block reduction. Deriving the block sum from it (via
        # select_one, as the non-split scan does) preserves operand order, which
        # is required for non-commutative combines -- tl.reduce may reorder
        # operands across lanes.
        block_scan = newvars(scan_shape)
        self.compute.writeline(
            f"{csv(block_scan)}= tl.associative_scan("
            f"({csv(broadcast_values)}), {dim}, {combine_helper_fn})"
        )
        block_sum = [
            cse_compute(
                f"triton_helpers.select_one({bscan}, "
                f"tl.arange(0, RBLOCK) == RBLOCK - 1, {dim}, keep_dims=False)",
                dtype=dtype,
                shape=reduced_shape,
            )
            for dtype, bscan in zip(dtypes, block_scan)
        ]

        exclusive_prefix = newvars(reduced_shape)
        pid = self.iteration_ranges_get_pid(self.range_trees[-1])
        if num_lanes == 2:
            self.compute.splice(
                f"""
                {exclusive_prefix[0]}, {exclusive_prefix[1]} = triton_helpers.exclusive_scan_decoupled_lookback_2(
                    {scratch_base},
                    {block_sum[0]},
                    {block_sum[1]},
                    {pid},
                    {combine_helper_fn},
                )
                """,
                strip=True,
            )
        elif element_nbits[0] == 64:
            self.compute.splice(
                f"""
                {exclusive_prefix[0]} = triton_helpers.exclusive_scan_decoupled_lookback_64(
                    {scratch_base},
                    {block_sum[0]},
                    {pid},
                    {combine_helper_fn},
                )
                """,
                strip=True,
            )
        else:
            if element_nbits[0] > 32:
                raise AssertionError(
                    f"expected element_nbits <= 32, got {element_nbits[0]}"
                )
            value_as_uint_dtype = f"tl.uint{element_nbits[0]}"
            self.compute.splice(
                f"""
                {exclusive_prefix[0]} = triton_helpers.exclusive_scan_decoupled_lookback(
                    {scratch_base},
                    {block_sum[0]},
                    {pid},
                    {combine_helper_fn},
                    DTYPE_VALUE_AS_UINT={value_as_uint_dtype},
                    DTYPE_PACK={scratch_type},
                )
                """,
                strip=True,
            )

        # combine_helper_fn returns a bare scalar for one lane and a tuple for
        # two, so the assignment target matches num_lanes.
        combined = newvars(scan_shape)
        combine_lhs = ", ".join(str(v) for v in combined)
        combine_args = ", ".join(str(v) for v in (*exclusive_prefix, *block_scan))
        self.compute.writeline(f"{combine_lhs} = {combine_helper_fn}({combine_args})")
        return tuple(
            cse_compute(
                f"tl.where(roffset == 0, {bscan}, {comb})",
                dtype=dtype,
                shape=scan_shape,
            )
            for dtype, bscan, comb in zip(dtypes, block_scan, combined)
        )

    def _get_heuristic(self):
        return "split_scan"

    def _get_grid_type(self) -> type[SplitScanGrid]:
        return SplitScanGrid
