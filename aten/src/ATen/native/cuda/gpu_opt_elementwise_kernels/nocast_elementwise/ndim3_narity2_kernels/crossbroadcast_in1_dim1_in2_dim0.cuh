// Cross-broadcast: IN1 is dim1-broadcast (stride11 == 0) and IN2 is
// dim0-broadcast (stride02 == 0); OUT is fully contiguous. IN1 depends only
// on (x, z), so its vt vector load is hoisted out of the dim1 row loop
// (stride11 == 0 makes the hoist legal); IN2 depends on (y, z), so one
// scalar is read per row and reused across all vt dim0 elements of that row.
// grid.z walks dim2 in z_t chunks, blockIdx.y walks dim1 rows in y_b chunks
// (tag p_e_gpu_3_2_xb).
template <int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0(
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f) {
  // Same-width 2/4-byte operands only (guarded by the caller): ub stays 1
  // and the per-z batches degenerate to the plain per-z code.
  constexpr int ub = (sizeof(res_t) == 16) ? 4 : 1;
  const int64_t z_base = static_cast<int64_t>(blockIdx.z) * z_t;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y_b = static_cast<int64_t>(blockDim.y);
  const int64_t y_base = static_cast<int64_t>(blockIdx.y) * y_b;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  const int64_t x_out = x * stride00;
  const char* in1_x = data1 + x * stride01;  // stride11 == 0: no y dependence
  for (int z_i = static_cast<int>(threadIdx.y) * ub; z_i < z_t;
       z_i += static_cast<int>(blockDim.y) * ub) {
    const int64_t z0 = z_base + z_i;
    if (z0 >= size2) {
      break;
    }
    // IN1 dim1-broadcast: hoist the vt load out of the row loop; IN1 is
    // contiguous on dim0 (stride01 == es) and its dim2 offset is vector
    // aligned (caller checked stride21 % al == 0).
    vec_arg0 in1[ub];
#pragma unroll
    for (int uu = 0; uu < ub; uu++) {
      if (z0 + uu < size2) {
        in1[uu] = *reinterpret_cast<const vec_arg0*>(in1_x + (z0 + uu) * stride21);
      }
    }
    char* out_z = data0 + x_out + z0 * stride20;
    #pragma unroll 2
    for (int64_t ty = 0; ty < y_b; ty++) {
      const int64_t yy = y_base + ty;
      if (yy >= size1) {
        break;
      }
      char* row_out = out_z + yy * stride10;
      // IN2 dim0-broadcast: one scalar per (y, z) row shared by all vt
      // dim0 elements.
      arg1_t in2[ub];
#pragma unroll
      for (int uu = 0; uu < ub; uu++) {
        if (z0 + uu < size2) {
          in2[uu] = c10::load(reinterpret_cast<const arg1_t*>(
              data2 + yy * stride12 + (z0 + uu) * stride22));
        }
      }
#pragma unroll
      for (int uu = 0; uu < ub; uu++) {
        if (z0 + uu < size2) {
          vec_res out;
#pragma unroll
          for (int i = 0; i < vt; i++) {
            out.val[i] = f(in1[uu].val[i], in2[uu]);
          }
          *reinterpret_cast<vec_res*>(row_out + uu * stride20) = out;
        }
      }
    }
  }
}


template <int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0(
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
  // Caller guarantees size0 % vt == 0, size1 <= 65535 * 8 (gridDim.y limit)
  // and size2 <= 65535 * z_t (gridDim.z limit).
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b = std::min<int64_t>(8, size1);
  dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  const int64_t grid_y = (size1 + y_b - 1) / y_b;
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>((size2 + z_t - 1) / z_t));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0<vt, z_t, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
