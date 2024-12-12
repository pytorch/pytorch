# mypy: allow-untyped-defs
import functools
from typing import Dict

import sympy

from torch._inductor import config
from torch._inductor.codegen.simd import IterationRangesRoot, prefix_is_reduction
from torch._inductor.codegen.triton import triton_compute_type, TritonKernel
from torch._inductor.runtime.triton_heuristics import split_scan_grid
from torch.utils._sympy.functions import CeilDiv

from ..utils import sympy_product


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
        tiling: Dict[str, sympy.Expr],
        pid_cache=None,
        fixed_config=None,
        **kwargs,
    ) -> None:
        assert pid_cache is None, "not supported"
        assert fixed_config is None, "not supported"
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
        assert len(self.numels) <= len(
            prefixes
        ), "z dimension not supported for split scan"
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
                    self,
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
        import triton.language as tl

        (dtype,) = dtypes
        (value,) = values

        compute_type = triton_compute_type(dtype)
        compute_type_triton = getattr(tl, compute_type[3:])

        element_nbits = compute_type_triton.primitive_bitwidth

        scratch_type = "tl.uint32" if element_nbits <= 16 else "tl.uint64"
        scratch_type_triton = getattr(tl, scratch_type[3:])
        scratch_elems_per_block = 3 if element_nbits == 64 else 1
        scratch_nbytes_per_block = scratch_elems_per_block * (
            scratch_type_triton.primitive_bitwidth // 8
        )

        cse_load = functools.partial(self.cse.generate, self.loads, dtype=dtype)
        cse_compute = functools.partial(self.cse.generate, self.compute)

        assert len(self.numels) == 2, "Unexpected tiling"
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
        scratch_base, offset = self.args.workspace(nbytes=nbytes, zero_fill=True)
        if offset != 0:
            scratch_base = cse_load(f"{scratch_base} + {self.index_to_str(offset)}")
        runtime_rblocks = cse_load(f"tl.num_programs({self.range_trees[-1].index})")
        scratch_base = cse_load(
            f"{scratch_base}.to(tl.pointer_type({scratch_type})) + xoffset * "
            f"{scratch_elems_per_block} * {runtime_rblocks}"
        )

        masks = {f"{tree.prefix}mask" for tree in self.range_trees}
        self.filter_masks(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"

        value = cse_compute(
            f"{value}.to({compute_type})",
            dtype=dtype,
        )
        value = cse_compute(
            f"tl.broadcast_to({value}, {self.dense_size_str()})",
            dtype=dtype,
        )

        combine_helper_fn = self._lift_helper(combine_fn, 1, (dtype,))
        dim = self.triton_tensor_ndim() - 1
        assert dim == 0, ""

        block_sum = cse_compute(
            f"tl.reduce({value}, {dim}, {combine_helper_fn})",
            dtype=dtype,
        )
        exclusive_prefix = self.cse.newvar(
            dtype=dtype,
        )
        if element_nbits == 64:
            self.compute.splice(
                f"""
                {exclusive_prefix} = triton_helpers.exclusive_scan_decoupled_lookback_64(
                    {scratch_base},
                    {block_sum},
                    {self.iteration_ranges_get_pid(self.range_trees[-1])},
                    {combine_helper_fn},
                )
                """,
                strip=True,
            )

        else:
            assert element_nbits <= 32
            value_as_uint_dtype = f"tl.uint{element_nbits}"

            self.compute.splice(
                f"""
                {exclusive_prefix} = triton_helpers.exclusive_scan_decoupled_lookback(
                    {scratch_base},
                    {block_sum},
                    {self.iteration_ranges_get_pid(self.range_trees[-1])},
                    {combine_helper_fn},
                    DTYPE_VALUE_AS_UINT={value_as_uint_dtype},
                    DTYPE_PACK={scratch_type},
                )
                """,
                strip=True,
            )
        # Compute final cumsum
        block_scan = cse_compute(
            f"tl.associative_scan({value}, {dim}, {combine_helper_fn})",
            dtype=dtype,
        )
        combined_result = cse_compute(
            f"{combine_helper_fn}({exclusive_prefix}, {block_scan})",
            dtype=dtype,
        )
        return (
            cse_compute(
                f"tl.where(roffset == 0, {block_scan}, {combined_result})",
                dtype=dtype,
            ),
        )

    def _get_heuristic(self):
        return "split_scan"

    def _get_grid_fn_str(self):
        return "split_scan_grid"

    def _get_grid_fn(self):
        return split_scan_grid
