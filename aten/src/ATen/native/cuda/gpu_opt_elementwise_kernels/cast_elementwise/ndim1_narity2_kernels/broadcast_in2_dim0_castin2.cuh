// div (60,) with a Long scalar denominator: OUT float
// contiguous, IN1 float contiguous, IN2 stride-0 (int64 scalar) cast on
// load. IN2 is read with one scalar load and converted once per thread;
// IN1/OUT use vt float vectors. tag p_e_gpu_1_2_cast_cb2.
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride10,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using vec_res = at::native::memory::aligned_vector<float, vt>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  // IN2 dim0-broadcast: one scalar read + convert per thread (stride == 0).
  const int64_t in2 = c10::load(reinterpret_cast<const int64_t*>(data2));
  const f_arg1_t a1 = c10::convert<f_arg1_t>(in2);
  const vec_res in1 = *reinterpret_cast<const vec_res*>(data1 + x * stride10);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(c10::convert<f_arg0_t>(in1.val[i]), a1);
  }
  *reinterpret_cast<vec_res*>(data0 + x * stride00) = out;
}


template <int nt, int vt, typename func_t>
static void launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride10,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2<nt, vt, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, data2, stride00, stride10, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
