
template <int nt, int vt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1_contiguous_alldim0(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    func_t f) {
  // ndim = 5, arity = 1. Output fully contiguous, input contiguous on dim0;
  // input's higher-dim strides are arbitrary multiples of the vector width
  // (0 for a broadcast dim).
  int64_t linear_idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (linear_idx >= N) {
    return;
  }
  const int64_t idx0 = linear_idx % size0;
  const int64_t idx1 = (linear_idx / size0) % size1;
  const int64_t idx2 = (linear_idx / (size0 * size1)) % size2;
  const int64_t idx3 = (linear_idx / (size0 * size1 * size2)) % size3;
  const int64_t idx4 = linear_idx / (size0 * size1 * size2 * size3);
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10 + idx2 * stride20 + idx3 * stride30 + idx4 * stride40;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11 + idx2 * stride21 + idx3 * stride31 + idx4 * stride41;

  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  LoadT ld = *reinterpret_cast<const LoadT*>(data1 + offset1);
  StoreT st;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    st.val[i] = f(ld.val[i]);
  }
  *reinterpret_cast<StoreT*>(data0 + offset0) = st;
}


template <int nt, int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_5_1_contiguous_alldim0(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1_contiguous_alldim0<nt, vt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, size3, size4,
          stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
