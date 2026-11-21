
// C3 patterns (pow on (32,)/(64,)/(128,)): OUT and IN2 contiguous with the
// same dtype (float or double), IN1 stride-0 (Long or Double, cast on load).
// tag p_e_gpu_1_2_cb.
template <int nt, int vt, typename res_t, typename LoadT, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride20,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  const LoadT in1 = c10::load(reinterpret_cast<const LoadT*>(data1));  // stride == 0
  const vec_res in2 = *reinterpret_cast<const vec_res*>(data2 + x * stride20);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(c10::convert<f_arg0_t>(in1), c10::convert<f_arg1_t>(in2.val[i]));
  }
  *reinterpret_cast<vec_res*>(data0 + x * stride00) = out;
}


template <int nt, int vt, typename res_t, typename LoadT, typename func_t>
static void launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride20,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, vt, res_t, LoadT, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, data2, stride00, stride20, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
