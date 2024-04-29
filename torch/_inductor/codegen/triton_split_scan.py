import functools

from typing import Optional, Set

import torch._inductor.runtime.hints
from torch._inductor import config

from torch._inductor.codegen.triton import (
    IterationRangesRoot,
    triton_compute_type,
    TritonKernel,
)

from torch._prims_common import prod

from torch.utils._sympy.functions import CeilDiv


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
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        reduction_hint=torch._inductor.runtime.hints.ReductionHint.DEFAULT,
        min_elem_per_thread=0,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache=None,
            reduction_hint=reduction_hint,
            min_elem_per_thread=min_elem_per_thread,
        )
        self.no_x_dim = True

    def initialize_range_tree(self, pid_cache):
        prefixes = "yxr"
        assert len(self.numels) <= len(
            prefixes
        ), "z dimension not supported for split scan"
        active_prefixes = prefixes[len(prefixes) - len(self.numels) :]

        grid_dims = "rxy"
        for numel, prefix in zip(self.numels, active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = 0 if is_reduction else None
            grid_dim = grid_dims.find(prefix)
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
        for tree in self.range_trees:
            tree.codegen_header(self.body)

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

        cse_load = functools.partial(self.cse.generate, self.loads)
        cse_compute = functools.partial(self.cse.generate, self.compute)

        assert len(self.numels) == 2, "Unexpected tiling"
        min_rblock = config.triton.min_split_scan_rblock
        max_blocks = prod(self.numels[:-1]) * CeilDiv(self.numels[-1], min_rblock)
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
        masks = sorted(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"

        value = cse_compute(f"{value}.to({compute_type})")
        value = cse_compute(f"tl.broadcast_to({value}, {self.dense_size_str()})")

        combine_helper_fn = self._lift_helper(combine_fn, 1)
        dim = self.triton_tensor_ndim() - 1
        assert dim == 0, ""

        block_sum = cse_compute(f"tl.reduce({value}, {dim}, {combine_helper_fn})")
        exclusive_prefix = self.cse.newvar()
        if element_nbits == 64:
            self.compute.splice(
                f"""
                {exclusive_prefix} = triton_helpers.exclusive_scan_decoupled_lookback_64(
                    {scratch_base},
                    {block_sum},
                    {self.range_trees[-1].get_pid()},
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
                    {self.range_trees[-1].get_pid()},
                    {combine_helper_fn},
                    DTYPE_VALUE_AS_UINT={value_as_uint_dtype},
                    DTYPE_PACK={scratch_type},
                )
                """,
                strip=True,
            )
        # Compute final cumsum
        block_scan = cse_compute(
            f"tl.associative_scan({value}, {dim}, {combine_helper_fn})"
        )
        combined_result = cse_compute(
            f"{combine_helper_fn}({exclusive_prefix}, {block_scan})"
        )
        return (
            cse_compute(f"tl.where(roffset == 0, {block_scan}, {combined_result})"),
        )

    def _get_heuristic(self):
        return "split_scan"

    def _get_grid_fn(self):
        return "split_scan_grid"
