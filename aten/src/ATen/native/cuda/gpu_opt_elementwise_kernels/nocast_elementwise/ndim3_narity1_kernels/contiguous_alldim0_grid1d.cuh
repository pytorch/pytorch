

// size0*size1 plane per block; gridDim.y carries size2 (caller guarantees
// size2 <= 65535). Used as the fallback when the 2D-grid kernel would waste

// try_launch_gpu_3_1).
template <int nt, int vt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1_contiguous_alldim0_grid1d(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    func_t f) {
  // ndim = 3, arity = 1. Output fully contiguous, input contiguous on dim0;
  // input's dim1/dim2 strides are arbitrary multiples of the vector width
  // (0 for a broadcast dim). blockIdx.y walks size2.
  const int64_t z = blockIdx.y;
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (idx >= size0 * size1) {
    return;
  }
  const int64_t idx0 = idx % size0;
  const int64_t idx1 = idx / size0;
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10 + z * stride20;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11 + z * stride21;

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
static void launch_elementwise_kernel_3_1_contiguous_alldim0_grid1d(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  // Caller guarantees size0 % vt == 0 and size2 <= 65535 (gridDim.y limit).
  const int64_t plane = size0 * size1;
  dim3 block(nt);
  dim3 grid(
      (plane + static_cast<int64_t>(block.x) * vt - 1) /
          (static_cast<int64_t>(block.x) * vt),
      size2);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_1_contiguous_alldim0_grid1d<nt, vt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
