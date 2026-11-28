
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_3_2_contiguous_alldim0_grid2d(
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

  // — blockIdx.x walks dim0 vectors, blockIdx.y walks dim1 rows in chunks of
  // y_b == blockDim.y, blockIdx.z walks dim2 (caller guards size2 <= 65535).
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (y >= size1) {
    return;
  }
  const int64_t z = blockIdx.z;
  const int64_t offset0 = x * stride00 + y * stride10 + z * stride20;
  const int64_t offset1 = x * stride01 + y * stride11 + z * stride21;
  const int64_t offset2 = x * stride02 + y * stride12 + z * stride22;

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

// For narrow dim0 (8 elements) fuse 16 dim2 planes into block.z; enabled only at
// the larger sizes where it measured faster, still calling the original func_t.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
__global__ void elementwise_kernel_3_2_contiguous_alldim0_grid2d_narrow_x(
    char* data0, char* data1, char* data2, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22, func_t f) {
  const int64_t y = threadIdx.y;
  const int64_t z = static_cast<int64_t>(blockIdx.z) * 16 + threadIdx.z;
  if (z >= size2) return;
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  const vec_arg0 a = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + z * stride21);
  const vec_arg1 b = *reinterpret_cast<const vec_arg1*>(data2 + y * stride12 + z * stride22);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; ++i) out.val[i] = f(a.val[i], b.val[i]);
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + z * stride20) = out;
}


template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_contiguous_alldim0_grid2d(
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
  if (size0 == 8 && size1 == 8 && vt == 8 && N >= 262144) {
    dim3 block(1, 8, 16);
    dim3 grid(1, 1, (size2 + 15) / 16);
    auto stream = at::cuda::getCurrentCUDAStream();
    elementwise_kernel_3_2_contiguous_alldim0_grid2d_narrow_x<vt, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(data0, data1, data2, size2,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Caller guarantees size0 % vt == 0, size1 <= 65535 * 8, size2 <= 65535.
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b = std::min<int64_t>(8, size1);
  dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  const int64_t grid_y = (size1 + y_b - 1) / y_b;
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>(size2));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_contiguous_alldim0_grid2d<vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
