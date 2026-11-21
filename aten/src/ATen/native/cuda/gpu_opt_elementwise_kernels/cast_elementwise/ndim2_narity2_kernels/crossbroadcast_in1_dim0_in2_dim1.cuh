
// C6 patterns ((21,1024), (22,1024)): OUT double fully contiguous, IN1 Long
// dim0-broadcast (stride01 == 0) + dim1-contiguous (cast on load), IN2 double
// dim1-broadcast (stride12 == 0) + dim0-contiguous. grid = (ceil(size0/(nt*vt)),
// size1); per row one scalar IN1 load + vt IN2 loads. tag p_e_gpu_2_2_cbx.
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1(
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
  using vec_res = at::native::memory::aligned_vector<double, vt>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  const int64_t in1 = c10::load(reinterpret_cast<const int64_t*>(data1 + y * stride11));  // stride01 == 0
  const vec_res in2 = *reinterpret_cast<const vec_res*>(data2 + x * stride02);  // stride12 == 0
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(c10::convert<f_arg0_t>(in1), in2.val[i]);
  }
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
}


template <int nt, int vt, typename func_t>
static void launch_cast_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1(
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
  cast_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, vt, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
