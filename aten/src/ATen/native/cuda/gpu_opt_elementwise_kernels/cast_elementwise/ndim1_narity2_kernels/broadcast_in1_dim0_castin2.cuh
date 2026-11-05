
#include "../../utils/common.cuh"

// C3b patterns (mul (128,) with a Long exponent-like operand): OUT float
// contiguous, IN1 stride-0 (float scalar), IN2 contiguous with a different
// storage dtype (int64) cast on load. IN1 is read with a scalar load, IN2
// in per-16B chunks. tag p_e_gpu_1_2_cast_cb.
template <int nt, typename res_t, typename in1_t, typename in2_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin2(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride20,
    func_t f) {
  constexpr int vt0 = per_op_vector_width<res_t>();
  constexpr int vt2 = per_op_vector_width<in2_t>();
  constexpr int vt = vt0 > vt2 ? vt0 : vt2;
  constexpr int n0 = vt / vt0;
  constexpr int n2 = vt / vt2;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  using vec_in2 = at::native::memory::aligned_vector<in2_t, vt2>;
  using vec_res = at::native::memory::aligned_vector<res_t, vt0>;
  const in1_t in1 = c10::load(reinterpret_cast<const in1_t*>(data1));  // stride == 0
  const f_arg0_t a0 = c10::convert<f_arg0_t>(in1);
  f_arg1_t v2[vt];
#pragma unroll
  for (int j = 0; j < n2; j++) {
    const vec_in2 in = *reinterpret_cast<const vec_in2*>(data2 + (x + j * vt2) * stride20);
#pragma unroll
    for (int i = 0; i < vt2; i++) {
      v2[j * vt2 + i] = c10::convert<f_arg1_t>(in.val[i]);
    }
  }
  res_t vals[vt];
#pragma unroll
  for (int i = 0; i < vt; i++) {
    vals[i] = f(a0, v2[i]);
  }
#pragma unroll
  for (int j = 0; j < n0; j++) {
    vec_res out;
#pragma unroll
    for (int i = 0; i < vt0; i++) {
      out.val[i] = vals[j * vt0 + i];
    }
    *reinterpret_cast<vec_res*>(data0 + (x + j * vt0) * stride00) = out;
  }
}


template <int nt, typename res_t, typename in1_t, typename in2_t, typename func_t>
static void launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin2(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t stride00, int64_t stride20,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int vt0 = per_op_vector_width<res_t>();
  constexpr int vt2 = per_op_vector_width<in2_t>();
  constexpr int vt = vt0 > vt2 ? vt0 : vt2;
  dim3 block(nt);
  dim3 grid((N + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin2<nt, res_t, in1_t, in2_t, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, data2, stride00, stride20, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
