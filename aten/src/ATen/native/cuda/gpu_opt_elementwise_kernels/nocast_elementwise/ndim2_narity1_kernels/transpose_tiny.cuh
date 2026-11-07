

// OUT dim0-contiguous; threads are linearized over OUT, so reads along IN
// and writes along OUT dim0 (stride00) are both element-contiguous — no
// shared-memory tile is needed for tiles this small, and the legacy path's
// full divmod offset computation is skipped entirely. Both the identity
// form (IN dim0-contiguous) and the reverse-transpose form (IN
// dim1-contiguous, IN is a transposed view of OUT) are plain elementwise
// copies in iterator coordinates: offset1 = i*stride01 + j*stride11.
// tag p_e_gpu_2_1_tp_tiny.
template <int nt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_transpose_tiny(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x;
  if (idx >= N) {
    return;
  }
  // i = out dim0, j = out dim1.
  const int64_t i = idx % size0;
  const int64_t j = idx / size0;
  const int64_t offset0 = i * stride00 + j * stride10;
  const int64_t offset1 = i * stride01 + j * stride11;
  *reinterpret_cast<res_t*>(data0 + offset0) =
      f(c10::load(reinterpret_cast<const arg0_t*>(data1 + offset1)));
}


template <int nt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_2_1_transpose_tiny(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + nt - 1) / nt);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1_transpose_tiny<nt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
