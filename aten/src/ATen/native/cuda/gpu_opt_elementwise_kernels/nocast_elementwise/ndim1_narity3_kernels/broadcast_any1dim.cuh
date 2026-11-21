
// Generalized 1D arity-3 broadcast: exactly one of IN1/IN2/IN3 is stride-0
// (scalar read), the other two are fully contiguous Float (vector read).
// broadcast_pos (0/1/2) is the broadcast operand's position in the functor
// argument sequence; pointers are laid out as data1/data2 = the contiguous
// pair in their original relative order, data3 = the broadcast operand.
// Only the all-float arity-3 combination is instantiated. tag p_e_gpu_1_3_v.
template <int nt, int vt, int broadcast_pos, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_3_broadcast_any1dim(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using vec = at::native::memory::aligned_vector<float, vt>;
  const vec in1 = *reinterpret_cast<const vec*>(data1 + x * 4);
  const vec in2 = *reinterpret_cast<const vec*>(data2 + x * 4);
  const float in3 = c10::load(reinterpret_cast<const float*>(data3));  // stride 0
  vec out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if constexpr (broadcast_pos == 0) {
      out.val[i] = f(in3, in1.val[i], in2.val[i]);       // IN1 broadcast
    } else if constexpr (broadcast_pos == 1) {
      out.val[i] = f(in1.val[i], in3, in2.val[i]);       // IN2 broadcast
    } else {
      out.val[i] = f(in1.val[i], in2.val[i], in3);       // IN3 broadcast
    }
  }
  *reinterpret_cast<vec*>(data0 + x * 4) = out;
}


template <int nt, int vt, int broadcast_pos, typename func_t>
static void launch_elementwise_kernel_1_3_broadcast_any1dim(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_3_broadcast_any1dim<nt, vt, broadcast_pos, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, data2, data3, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
