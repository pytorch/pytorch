
// C4-1 patterns ((4096,512), (4096,3584)): OUT float fully contiguous, IN1
// BFloat16 fully contiguous (cast on load), IN2 float dim0-broadcast
// (stride02 == 0). grid = (ceil(size0/(nt*vt)), size1); per row one scalar
// IN2 load feeds vt IN1 loads. tag p_e_gpu_2_2_cbd0_2d.
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y = static_cast<int64_t>(blockIdx.y);
  using vec_res = at::native::memory::aligned_vector<float, vt>;
  using vec_in1 = at::native::memory::aligned_vector<c10::BFloat16, vt>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  const vec_in1 in1 = *reinterpret_cast<const vec_in1*>(data1 + y * stride11 + x * stride01);
  const float in2 = c10::load(reinterpret_cast<const float*>(data2 + y * stride12));  // stride02 == 0
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(c10::convert<f_arg0_t>(in1.val[i]), in2);
  }
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
}


template <int nt, int vt, typename func_t>
static void launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d(
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
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  dim3 block(nt);
  dim3 grid(grid_x, size1);
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d<nt, vt, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
