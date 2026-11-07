
// (3072,2,3294) add: OUT float fully contiguous, IN1 BFloat16
// dim0+dim1 contiguous + dim2-broadcast (stride21 == 0, cast on load), IN2
// float dim0-contiguous + dim1-broadcast (stride12 == 0) + dim2-contiguous.
// IN2's vt-wide float vector load is hoisted out of the y row loop
// (stride12 == 0 makes the hoist legal); IN1 is vector-read per row and is
// z-invariant (stride21 == 0). tag p_e_gpu_3_2_cbd1f.
template <int nt, int z_t, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f, int64_t y_t, int64_t y_remain) {
  const int64_t z_base = static_cast<int64_t>(blockIdx.z) * z_t;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;
  using vec_res = at::native::memory::aligned_vector<float, vt>;
  using vec_in1 = at::native::memory::aligned_vector<c10::BFloat16, vt>;
  using vec_in2 = at::native::memory::aligned_vector<float, vt>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  const int64_t x_out = x * stride00;
  for (int64_t z_i = 0; z_i < z_t && z_base + z_i < size2; z_i++) {
    const int64_t z = z_base + z_i;
    // IN2 is dim1-broadcast (stride12 == 0): one vector load feeds all y rows.
    const vec_in2 in2 = *reinterpret_cast<const vec_in2*>(data2 + x * stride02 + z * stride22);
    const int64_t z_out = z * stride20;
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      // IN1 is dim2-broadcast (stride21 == 0): z-invariant, load per row.
      const vec_in1 in1 = *reinterpret_cast<const vec_in1*>(data1 + y * stride11 + x * stride01);
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(c10::convert<f_arg0_t>(in1.val[i]), c10::convert<f_arg1_t>(in2.val[i]));
      }
      *reinterpret_cast<vec_res*>(data0 + z_out + y * stride10 + x_out) = out;
    }
  }
}


template <int nt, int z_t, int vt, typename func_t>
static void launch_cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1(
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
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y, (size2 + z_t - 1) / z_t);
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1<nt, z_t, vt, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12,
          stride20, stride21, stride22, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
