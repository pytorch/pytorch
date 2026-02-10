import functools

from torch._cutedsl._compile_with_safe_names import _compile_with_safe_names


@functools.cache
def _compile_scaled_grouped_mm_prepare_metadata():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    @cute.kernel
    def _scaled_grouped_mm_prepare_metadata_kernel(
        G: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        base_a_u64: cutlass.Int64,
        base_b_u64: cutlass.Int64,
        base_c_u64: cutlass.Int64,
        base_scale_a_u64: cutlass.Int64,
        base_scale_b_u64: cutlass.Int64,
        offs: cute.Tensor,
        sizeof_ab: cutlass.Int32,
        sizeof_scale_ab: cutlass.Int32,
        sizeof_c: cutlass.Int32,
        stride_a: tuple[cutlass.Int32, cutlass.Int32],
        stride_b: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        stride_c: tuple[cutlass.Int32, cutlass.Int32],
        stride_scale_a: tuple[cutlass.Int32, cutlass.Int32],
        stride_scale_b: tuple[cutlass.Int32, cutlass.Int32],
        CLUSTER_TILE_M: cutlass.Int32,
        CLUSTER_TILE_N: cutlass.Int32,
        out_mnkl: cute.Tensor,
        out_ptrs_abc: cute.Tensor,
        out_ptrs_scale_ab: cute.Tensor,
        out_strides_abc: cute.Tensor,
        out_nclusters: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdimx, _, _ = cute.arch.block_dim()
        g = bidx * bdimx + tidx

        if g < G:
            off_start = 0
            if g > 0:
                off_start = offs[g - 1]
            off_end = offs[g]
            group_size = off_end - off_start

            byte_off_a = (
                cutlass.Int64(off_start) * stride_a[0] * cutlass.Int64(sizeof_ab)
            )
            b_byte_off = cutlass.Int64(g) * stride_b[0] * cutlass.Int64(sizeof_ab)
            c_byte_off = (
                cutlass.Int64(off_start) * stride_c[0] * cutlass.Int64(sizeof_c)
            )

            # Scale factors for A are stored per group, but with each
            # group padded to 128 rows. Compute the scale row offset accordingly.
            off_start_scale_a = cutlass.Int32(0)
            for i in cutlass.range(G):
                if i < g:
                    off_start = 0
                    if i > 0:
                        off_start = offs[i - 1]
                    off_end = offs[i]
                    off_start_scale_a += cute.ceil_div(off_end - off_start, 128) * 128
            byte_off_scale_a = (
                cutlass.Int64(off_start_scale_a)
                * stride_scale_a[0]
                * cutlass.Int64(sizeof_scale_ab)
            )

            byte_off_scale_b = (
                cutlass.Int64(g) * stride_scale_b[0] * cutlass.Int64(sizeof_scale_ab)
            )

            out_mnkl[g, 0] = group_size
            out_mnkl[g, 1] = N
            out_mnkl[g, 2] = K
            out_mnkl[g, 3] = cutlass.Int32(1)

            out_ptrs_abc[g, 0] = base_a_u64 + byte_off_a
            out_ptrs_abc[g, 1] = base_b_u64 + b_byte_off
            out_ptrs_abc[g, 2] = base_c_u64 + c_byte_off
            out_ptrs_scale_ab[g, 0] = base_scale_a_u64 + byte_off_scale_a
            out_ptrs_scale_ab[g, 1] = base_scale_b_u64 + byte_off_scale_b

            out_strides_abc[g, 0, 0] = cutlass.Int32(stride_a[0])
            out_strides_abc[g, 0, 1] = cutlass.Int32(stride_a[1])
            out_strides_abc[g, 1, 0] = cutlass.Int32(stride_b[2])
            out_strides_abc[g, 1, 1] = cutlass.Int32(stride_b[1])
            out_strides_abc[g, 2, 0] = cutlass.Int32(stride_c[0])
            out_strides_abc[g, 2, 1] = cutlass.Int32(stride_c[1])

        if tidx == 0 and bidx == 0:
            nclusters = cutlass.Int32(0)
            for i in cutlass.range(G):
                off_start = 0
                if i > 0:
                    off_start = offs[i - 1]
                off_end = offs[i]
                nclusters_m = cute.ceil_div(off_end - off_start, CLUSTER_TILE_M)
                nclusters_n = cute.ceil_div(N, CLUSTER_TILE_N)
                nclusters += nclusters_m * nclusters_n
            out_nclusters[0] = nclusters

    @cute.jit
    def _launch_scaled_grouped_mm_prepare_metadata(
        G: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        base_a_u64: cutlass.Int64,
        base_b_u64: cutlass.Int64,
        base_c_u64: cutlass.Int64,
        base_scale_a_u64: cutlass.Int64,
        base_scale_b_u64: cutlass.Int64,
        offs: cute.Tensor,
        sizeof_ab: cutlass.Constexpr,
        sizeof_scale_ab: cutlass.Constexpr,
        sizeof_c: cutlass.Constexpr,
        stride_a: tuple[cutlass.Int32, cutlass.Int32],
        stride_b: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        stride_c: tuple[cutlass.Int32, cutlass.Int32],
        stride_scale_a: tuple[cutlass.Int32, cutlass.Int32],
        stride_scale_b: tuple[cutlass.Int32, cutlass.Int32],
        CLUSTER_TILE_M: cutlass.Int32,
        CLUSTER_TILE_N: cutlass.Int32,
        out_mnkl: cute.Tensor,
        out_ptrs_abc: cute.Tensor,
        out_ptrs_scale_ab: cute.Tensor,
        out_strides_abc: cute.Tensor,
        out_nclusters: cute.Tensor,
        num_blocks: cutlass.Int32,
        threads_per_block: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _scaled_grouped_mm_prepare_metadata_kernel(
            G,
            M,
            N,
            K,
            base_a_u64,
            base_b_u64,
            base_c_u64,
            base_scale_a_u64,
            base_scale_b_u64,
            offs,
            sizeof_ab,
            sizeof_scale_ab,
            sizeof_c,
            stride_a,
            stride_b,
            stride_c,
            stride_scale_a,
            stride_scale_b,
            CLUSTER_TILE_M,
            CLUSTER_TILE_N,
            out_mnkl,
            out_ptrs_abc,
            out_ptrs_scale_ab,
            out_strides_abc,
            out_nclusters,
        ).launch(
            grid=(num_blocks, 1, 1), block=(threads_per_block, 1, 1), stream=stream
        )

    g = cute.sym_int()
    fake_offs = make_fake_tensor(cutlass.Int32, (g,), stride=(1,))
    fake_ptrs_abc = make_fake_tensor(cutlass.Int64, (g, 3), stride=(3, 1))
    fake_ptrs_scale_ab = make_fake_tensor(cutlass.Int64, (g, 2), stride=(2, 1))
    fake_mnkl = make_fake_tensor(cutlass.Int32, (g, 4), stride=(4, 1))
    fake_strides_abc = make_fake_tensor(cutlass.Int32, (g, 3, 2), stride=(6, 2, 1))
    fake_nclusters = make_fake_tensor(cutlass.Int32, (1,), stride=(1,))
    fake_stream = make_fake_stream()

    compiled = _compile_with_safe_names(
        lambda: cute.compile(
            _launch_scaled_grouped_mm_prepare_metadata,
            G=0,
            M=0,
            N=0,
            K=0,
            base_a_u64=0,
            base_b_u64=0,
            base_c_u64=0,
            base_scale_a_u64=0,
            base_scale_b_u64=0,
            offs=fake_offs,
            sizeof_ab=1,
            sizeof_scale_ab=1,
            sizeof_c=2,
            stride_a=(cute.sym_int(), cute.sym_int()),
            stride_b=(cute.sym_int(), cute.sym_int(), cute.sym_int()),
            stride_c=(cute.sym_int(), cute.sym_int()),
            stride_scale_a=(cute.sym_int(), cute.sym_int()),
            stride_scale_b=(cute.sym_int(), cute.sym_int()),
            CLUSTER_TILE_M=0,
            CLUSTER_TILE_N=0,
            out_mnkl=fake_mnkl,
            out_ptrs_abc=fake_ptrs_abc,
            out_ptrs_scale_ab=fake_ptrs_scale_ab,
            out_strides_abc=fake_strides_abc,
            out_nclusters=fake_nclusters,
            num_blocks=1,
            threads_per_block=1,
            stream=fake_stream,
            options="--enable-assertions",
        )
    )
    return compiled
