
// The cast path (gpu_kernel_impl with dynamic casting) has its own dispatch:

// kernels live outside try_launch_gpu_kernel. IN is broadcast on dim1
// (stride11 == 0) and contiguous on dim0 (int64): each output element
// (idx0, idx1) reads in[idx0] and converts it to float. Only the
// Float <- Long combination is instantiated; every other dtype mix falls
// through to the legacy cast path.

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_2_1_broadcast_dim1(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f) {
  int64_t linear_idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (linear_idx >= N) {
    return;
  }
  const int64_t idx0 = linear_idx % size0;
  const int64_t idx1 = linear_idx / size0;
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11;  // stride11 == 0

  using LoadT = at::native::memory::aligned_vector<int64_t, vt>;
  using StoreT = at::native::memory::aligned_vector<float, vt>;
  LoadT ld = *reinterpret_cast<const LoadT*>(data1 + offset1);
  StoreT st;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    st.val[i] = f(c10::convert<f_arg0_t>(ld.val[i]));
  }
  *reinterpret_cast<StoreT*>(data0 + offset0) = st;
}


template <int nt, int vt, typename func_t>
static void launch_cast_elementwise_kernel_2_1_broadcast_dim1(
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
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_2_1_broadcast_dim1<nt, vt, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
