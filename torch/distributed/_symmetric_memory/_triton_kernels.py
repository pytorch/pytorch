from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_partials_first_dim_kernel(
        partials,
        out,
        shard_elems: tl.constexpr,
        group_size: tl.constexpr,
        do_avg: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < shard_elems
        acc = tl.zeros((BLOCK,), tl.float32)
        for rank in tl.static_range(0, group_size):
            vals = tl.load(
                partials + rank * shard_elems + offsets,
                mask=mask,
                other=0.0,
            )
            acc += vals.to(tl.float32)
        if do_avg:
            acc /= group_size
        # The accumulation stays in fp32 and is rounded exactly once, here.
        tl.store(out + offsets, acc.to(out.dtype.element_ty), mask=mask)
