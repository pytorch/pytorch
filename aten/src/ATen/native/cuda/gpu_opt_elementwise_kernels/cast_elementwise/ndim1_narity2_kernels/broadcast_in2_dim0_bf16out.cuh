// mul BFloat16 x Float scalar, e.g. [1152]/[524288]/
// [4194304]): OUT BFloat16 contiguous, IN1 BFloat16 contiguous, IN2
// stride-0 Float scalar read once per thread and converted to the functor's
// arg type. OUT converts on store from the functor's result type. The
// functor is BinaryFunctor<BFloat16,BFloat16,BFloat16,opmath> so IN1/OUT
// conversion is a no-op and only IN2 crosses storage types.
// tag p_e_gpu_1_2_cast_cb2_bf16.
template <int nt, int vt, typename res_st, typename arg0_st, typename arg1_st, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride10,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  using vec_in0 = at::native::memory::aligned_vector<arg0_st, vt>;
  using vec_out = at::native::memory::aligned_vector<res_st, vt>;
  // IN2 dim0-broadcast: one scalar read + convert per thread (stride == 0).
  const f_arg1_t a1 = c10::convert<f_arg1_t>(c10::load(reinterpret_cast<const arg1_st*>(data2)));
  const vec_in0 in1 = *reinterpret_cast<const vec_in0*>(data1 + x * stride10);
  vec_out out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = c10::convert<res_st>(f(c10::convert<f_arg0_t>(in1.val[i]), a1));
  }
  *reinterpret_cast<vec_out*>(data0 + x * stride00) = out;
}


template <int nt, int vt, typename res_st, typename arg0_st, typename arg1_st, typename func_t>
static void launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out(
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
  cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out<nt, vt, res_st, arg0_st, arg1_st, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, data2, stride00, stride10, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
