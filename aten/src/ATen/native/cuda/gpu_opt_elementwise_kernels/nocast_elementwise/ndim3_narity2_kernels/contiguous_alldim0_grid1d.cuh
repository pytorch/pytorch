

// full numel per block with a flat grid, so it has no gridDim.y/z limit;
// used as the fallback when size2 exceeds the 2D-grid gridDim.z limit

template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_contiguous_alldim0_grid1d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f) {
  // ndim = 3, arity = 2. Output fully contiguous, both inputs contiguous on
  // dim0; their higher-dim strides are arbitrary multiples of the vector
  // width (0 for a broadcast dim, which degrades to reading the same row
  // repeatedly - semantically correct for dim1/dim2 broadcast).
  int64_t linear_idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (linear_idx >= N) {
    return;
  }
  const int64_t idx0 = linear_idx % size0;
  const int64_t idx1 = (linear_idx / size0) % size1;
  const int64_t idx2 = linear_idx / (size0 * size1);
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10 + idx2 * stride20;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11 + idx2 * stride21;
  const int64_t offset2 = idx0 * stride02 + idx1 * stride12 + idx2 * stride22;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + offset1);
  vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + offset2);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(in1.val[i], in2.val[i]);
  }
  *reinterpret_cast<vec_res*>(data0 + offset0) = out;
}


template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_contiguous_alldim0_grid1d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_contiguous_alldim0_grid1d<nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
