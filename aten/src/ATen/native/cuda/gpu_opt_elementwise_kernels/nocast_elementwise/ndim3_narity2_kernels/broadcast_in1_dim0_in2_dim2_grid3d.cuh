
// 3D-grid division-free version of elementwise_kernel_3_2_broadcast_dim0:
// IN1 is dim0-broadcast (stride01 == 0) and depends on dim1; it also varies
// along dim2 unless stride21 == 0. IN2 depends on (dim0, dim1) and is
// z-invariant (stride22 == 0). Two compile-time instances via kIn1ZInvariant
// keep the hot path branch-free: with stride21 == 0 the functor output is
// computed once per thread and reused for all z_t rows (same code as the
// original single-path kernel); with stride21 != 0 IN1 is reloaded per z row.
// Either way each thread issues z_t vector stores and the addressing stays
// div/mod-free.
// C1: 2D block (blockDim.x == nt_x, blockDim.y == y_b) — blockIdx.y walks dim1
// rows in chunks of y_b, and a y batch loop covers y_blocks > 65535 (the
// gridDim.y hardware limit), so size1 is unbounded. The z store loop stays
// inside the y loop so the per-(x vector, y row) work — the IN2 vector load,
// and the functor evaluation when stride21 == 0 — is shared by all z_t rows.
// C1b: nt_x is min(size0 / vt, 128) — a fixed 128 wastes 127/128 thread
// slots when size0 is small (e.g. size0 == 8, vt == 8); y_b then lifts the
// block to at least one full warp.
template <bool kIn1ZInvariant, int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t,
          typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f, int64_t z_t_, int64_t z_remain) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y_b = static_cast<int64_t>(blockDim.y);
  const int64_t y_blocks = (size1 + y_b - 1) / y_b;
  int64_t z_loop = z_t_;
  if (z_remain != 0 && blockIdx.z == (gridDim.z - 1)) {
    z_loop = z_remain;
  }
  const int64_t z_start = static_cast<int64_t>(blockIdx.z) * z_t_;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  for (int64_t yb = static_cast<int64_t>(blockIdx.y); yb < y_blocks;
       yb += static_cast<int64_t>(gridDim.y)) {
    const int64_t y = yb * y_b + static_cast<int64_t>(threadIdx.y);
    if (y >= size1) {
      continue;
    }
    vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + y * stride12 + x * stride02);
    if constexpr (kIn1ZInvariant) {
      // IN1 is z-invariant: evaluate once per (x vector, y row) and reuse
      // the result for all z_t rows of this thread.
      const arg0_t in1 = c10::load(reinterpret_cast<const arg0_t*>(data1 + y * stride11));
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(in1, in2.val[i]);
      }
      for (int64_t z_idx = 0; z_idx < z_loop; z_idx++) {
        const int64_t z = z_start + z_idx;
        const int64_t offset0 = y * stride10 + z * stride20 + x * stride00;
        *reinterpret_cast<vec_res*>(data0 + offset0) = out;
      }
    } else {
      // IN1 varies along dim2: reload it per z row.
      for (int64_t z_idx = 0; z_idx < z_loop; z_idx++) {
        const int64_t z = z_start + z_idx;
        const arg0_t in1 = c10::load(
            reinterpret_cast<const arg0_t*>(data1 + y * stride11 + z * stride21));
        vec_res out;
#pragma unroll
        for (int i = 0; i < vt; i++) {
          out.val[i] = f(in1, in2.val[i]);
        }
        const int64_t offset0 = y * stride10 + z * stride20 + x * stride00;
        *reinterpret_cast<vec_res*>(data0 + offset0) = out;
      }
    }
  }
}


template <int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  // C1b: cap nt_x at x_vec (small size0 must not waste thread slots) and
  // lift y_b so the block keeps at least one full warp.
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b = std::min<int64_t>(
      std::max<int64_t>(8, (32 + nt_x - 1) / nt_x), size1);
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  // C1: chunk dim1 rows across blockDim.y lanes; the y batch loop in the
  // kernel covers y_blocks > 65535, so grid_y stays within the hardware limit.
  const int64_t y_blocks = (size1 + y_b - 1) / y_b;
  // Keep the grid_y block count at the historical 65535 cap: beyond that the
  // per-block scheduling overhead dominates (measured +31% at 131072 blocks
  // vs 65535, bf16 3_2_bd0_3d 64M). Never exceed the device gridDim.y limit.
  const int64_t grid_y = std::min<int64_t>(
      y_blocks, gpu_grid_cap(1));
  const int64_t grid_z = (size2 + z_t - 1) / z_t;
  const int64_t z_remain = size2 - (grid_z - 1) * z_t;
  dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>(grid_z));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Pick the compile-time instance from the runtime stride21 on the host side:
  // each device instance stays branch-free.
  if (stride21 == 0) {
    elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<true, vt, z_t, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(
            N, data0, data1, data2, size0, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, z_t, z_remain);
  } else {
    elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<false, vt, z_t, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(
            N, data0, data1, data2, size0, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, z_t, z_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
