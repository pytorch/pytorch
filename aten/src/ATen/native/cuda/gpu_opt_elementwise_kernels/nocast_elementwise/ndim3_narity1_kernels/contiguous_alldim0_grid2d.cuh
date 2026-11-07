
// Contiguous on dim0 with a dim2 batch axis: per thread, one vector load on
// dim0 per (row, z slice), with a grid-stride loop over dim1 rows and a chunk
// of z_t dim2 slices (ILP along the batch axis; gridDim.z stays within the
// 65535 hardware limit for any size2 <= 65535).
//
// Block/grid derivation is workload-driven, not dim0-driven: the block is
// filled toward the kernel launch bound (1024 threads) instead of being
// sized from dim0 alone, grid.y is compressed to a cap only when the
// uncompressed grid would still saturate the device (see the launcher), and
// the z_t chunk is picked adaptively so small shapes do not collapse to a
// single block (16K would otherwise launch grid=(1,1,1)).
// Small grids keep the one-row-per-thread layout, so scheduling pressure
// never exceeds the pre-change layout.
template <int vt, int z_t, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_3_1_contiguous_alldim0_grid2d(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    func_t f) {

  // contiguous on dim0; input's dim1/dim2 strides are arbitrary multiples of
  // the vector width (0 for a broadcast dim).

  // blockIdx.x walks dim0 vectors, blockIdx.y walks dim1 rows with a
  // grid-stride loop (gridDim.y may be compressed by the caller), blockIdx.z
  // walks dim2 in chunks of z_t (caller guards size2 <= 65535).
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t z_base = static_cast<int64_t>(blockIdx.z) * z_t;

  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  for (int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       y < size1;
       y += static_cast<int64_t>(gridDim.y) * blockDim.y) {
    const int64_t offset0 = x * stride00 + y * stride10;
    const int64_t offset1 = x * stride01 + y * stride11;
#pragma unroll 2
    for (int z_i = 0; z_i < z_t && z_base + z_i < size2; z_i++) {
      const int64_t z = z_base + z_i;
      LoadT ld = *reinterpret_cast<const LoadT*>(data1 + offset1 + z * stride21);
      StoreT st;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        st.val[i] = f(ld.val[i]);
      }
      *reinterpret_cast<StoreT*>(data0 + offset0 + z * stride20) = st;
    }
  }
}


template <int vt, int z_t, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d_impl(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    const func_t& f) {
  if (N == 0) {
    return;
  }
  // Caller guarantees size0 % vt == 0, size1 <= 65535 * 8, size2 <= 65535.
  // Workload-driven layout: fill the block toward the launch bound instead
  // of sizing it from dim0 alone, walk dim2 in z_t chunks per thread, and
  // compress grid.y to a cap only when the uncompressed grid still
  // saturates the device (per-thread work grows instead of the grid).
  constexpr int64_t k_target_block = 1024;  // C10_LAUNCH_BOUNDS_2(1024, 4)
  constexpr int64_t k_grid_y_cap = 8;
  constexpr int64_t k_min_full_grid = 2048;
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b =
      std::min<int64_t>(std::max<int64_t>(k_target_block / nt_x, 1), size1);
  dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  const int64_t grid_y_full = (size1 + y_b - 1) / y_b;
  const int64_t grid_z = (size2 + z_t - 1) / z_t;
  const int64_t full_grid = grid_x * grid_y_full * grid_z;
  const int64_t grid_y = full_grid >= k_min_full_grid
                             ? std::min<int64_t>(grid_y_full, k_grid_y_cap)
                             : grid_y_full;
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>(grid_z));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_1_contiguous_alldim0_grid2d<vt, z_t, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// z_t is workload-adaptive: the z_t = 8 layout compresses the block count by
// 8x, which under-fills the device on small shapes (16K measures grid=(1,1,1)
// - a single block - at 0.63x the unfolded layout). Unfold z until the grid
// is wide enough to schedule, then keep the z_t = 8 ILP (at 4M+ the unfolded
// layouts regress 1.25-1.48x). Thresholds from the 3_1_d0c_2d sweep (bf16 +
// fp32, 16K-16M): blocks < 16 -> z_t 1 (16K-256K, 1.55-1.85x); blocks < 64 ->
// z_t 4 (1M-2M, best of both); else z_t 8 (2M+ where it wins outright).
template <int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int64_t k_target_block = 1024;
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b =
      std::min<int64_t>(std::max<int64_t>(k_target_block / nt_x, 1), size1);
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  const int64_t grid_y_full = (size1 + y_b - 1) / y_b;
  const int64_t blocks8 = grid_x * grid_y_full * ((size2 + 7) / 8);
  if (blocks8 < 16) {
    launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d_impl<vt, 1, res_t, arg0_t>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (blocks8 < 64) {
    launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d_impl<vt, 4, res_t, arg0_t>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else {
    launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d_impl<vt, 8, res_t, arg0_t>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  }
}
