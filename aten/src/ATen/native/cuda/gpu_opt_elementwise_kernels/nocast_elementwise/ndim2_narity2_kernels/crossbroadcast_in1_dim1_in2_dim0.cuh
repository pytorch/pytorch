
// Mirror of crossbroadcast_in1_dim0_in2_dim1 -- arg0 dim1-broadcast (stride11 == 0)
// with a vector read hoisted out of the row loop, arg1 dim0-broadcast
// (stride02 == 0) read as one scalar per row. tag p_e_gpu_2_2_xl.
template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_crossbroadcast_in1_dim1_in2_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    func_t f, int64_t y_t, int64_t y_remain) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  // IN1 dim1-broadcast: read once per (x vector), reused across all rows.
  const vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + x * stride01);  // stride11 == 0
  #pragma unroll 2
  for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
    const int64_t y = y_start + y_idx;
    // IN2 dim0-broadcast: one scalar per row, independent of x.
    const arg1_t in2 = c10::load(reinterpret_cast<const arg1_t*>(data2 + y * stride12));  // stride02 == 0
    vec_res out;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      out.val[i] = f(in1.val[i], in2);
    }
    *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
  }
}


template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_2_2_crossbroadcast_in1_dim1_in2_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  // Grow the per-block row tile so gridDim.y never exceeds the device limit:
  // size1 is unbounded (e.g. a coalesced [large, small] broadcast). The kernel
  // handles any y_t through its y_remain tail, and y_t stays 8 for normal
  // sizes, so the measured win domain is unchanged.
  const int64_t grid_cap_y = gpu_grid_cap(1);
  int64_t y_t = 8;
  if (grid_cap_y > 0) {
    const int64_t min_y_t = (size1 + grid_cap_y - 1) / grid_cap_y;
    if (min_y_t > y_t) {
      y_t = min_y_t;
    }
  }
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_2_crossbroadcast_in1_dim1_in2_dim0<nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
