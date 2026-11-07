// compare/copy (8,) gathers with IN1 stride 96/100: OUT
// contiguous, IN1 non-contiguous strided gather. One element per thread;
// IN1 uses an element-wise c10::load (no vectorization is possible across a
// strided source), OUT stores one element at its contiguous offset. The
// caller passes byte-unit strides (TensorIteratorBase::strides); both
// offsets apply them directly on char* data. tag p_e_gpu_1_1_s.
template <int nt, int work, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_1_strided(
    int64_t N,
    char* data0, char* data1,
    int64_t stride00, int64_t stride01,
    func_t f) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * nt * work + threadIdx.x;
  arg0_t inputs[work];
#pragma unroll
  for (int i = 0; i < work; ++i) {
    const int64_t idx = base + i * nt;
    if (idx < N) {
      inputs[i] = c10::load(reinterpret_cast<const arg0_t*>(data1 + idx * stride01));
    }
  }
#pragma unroll
  for (int i = 0; i < work; ++i) {
    const int64_t idx = base + i * nt;
    if (idx < N) {
      *reinterpret_cast<res_t*>(data0 + idx * stride00) = f(inputs[i]);
    }
  }
}


template <int nt, typename res_t, typename arg0_t, typename func_t>
static void launch_gpu_1_1_strided(
    int64_t N,
    char* data0, char* data1,
    int64_t stride00, int64_t stride01,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N > 262144) {
    dim3 grid((N + static_cast<int64_t>(nt) * 4 - 1) / (static_cast<int64_t>(nt) * 4));
    elementwise_kernel_1_1_strided<nt, 4, res_t, arg0_t, func_t>
        <<<grid, block, 0, stream>>>(N, data0, data1, stride00, stride01, f);
  } else {
    dim3 grid((N + static_cast<int64_t>(nt) - 1) / static_cast<int64_t>(nt));
    elementwise_kernel_1_1_strided<nt, 1, res_t, arg0_t, func_t>
        <<<grid, block, 0, stream>>>(N, data0, data1, stride00, stride01, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
