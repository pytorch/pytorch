
// 3D-grid division-free version of elementwise_kernel_3_2_broadcast_arg1_dim0:
// blockIdx.x walks dim0 vectors, blockIdx.y indexes dim1 (the only dim IN2
// depends on), blockIdx.z walks dim2 rows in chunks of z_t. IN2 is loaded
// once per thread and reused across the z_t row loop; no int64 div/mod.
template <int nt, int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_in2_dim02_grid3d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f, int64_t z_t_, int64_t z_remain) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y = static_cast<int64_t>(blockIdx.y);
  int64_t z_loop = z_t_;
  if (z_remain != 0 && blockIdx.z == (gridDim.z - 1)) {
    z_loop = z_remain;
  }
  const int64_t z_start = static_cast<int64_t>(blockIdx.z) * z_t_;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  // IN2 depends on dim1 only (dim0/dim2 broadcast): one scalar load.
  const arg1_t in2 = c10::load(reinterpret_cast<const arg1_t*>(data2 + y * stride12));
  #pragma unroll 2
  for (int64_t z_idx = 0; z_idx < z_loop; z_idx++) {
    const int64_t z = z_start + z_idx;
    const int64_t offset0 = y * stride10 + z * stride20 + x * stride00;
    const int64_t offset1 = y * stride11 + z * stride21 + x * stride01;
    vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + offset1);
    vec_res out;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      out.val[i] = f(in1.val[i], in2);
    }
    *reinterpret_cast<vec_res*>(data0 + offset0) = out;
  }
}

// For size0=64, vt=8 map the original serial z loop to block.y, preserving reuse
// of the dim1-broadcast scalar input and computing each output via the real func_t.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
__global__ void elementwise_kernel_3_2_broadcast_in2_dim02_grid3d_narrow_x(
    char* data0, char* data1, char* data2, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride10, int64_t stride11,
    int64_t stride12, int64_t stride20, int64_t stride21, func_t f) {
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  __shared__ arg1_t in2;
  if (threadIdx.x == 0 && threadIdx.y == 0)
    in2 = c10::load(reinterpret_cast<const arg1_t*>(data2 + blockIdx.y * stride12));
  __syncthreads();
  const int64_t x = static_cast<int64_t>(threadIdx.x) * vt;
  const int64_t y = blockIdx.y;
  const int64_t z = static_cast<int64_t>(blockIdx.z) * 8 + threadIdx.y;
  if (z >= size2) return;
  const vec_arg0 a = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + z * stride21 + x * stride01);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; ++i) out.val[i] = f(a.val[i], in2);
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + z * stride20 + x * stride00) = out;
}


template <int nt, int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid3d(
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
  if (size0 == 64 && vt == 8) {
    dim3 block(8, 8);
    dim3 grid(1, size1, (size2 + 7) / 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    elementwise_kernel_3_2_broadcast_in2_dim02_grid3d_narrow_x<vt, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(data0, data1, data2, size1, size2,
            stride00, stride01, stride10, stride11, stride12, stride20, stride21, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = size1;
  const int64_t grid_z = (size2 + z_t - 1) / z_t;
  const int64_t z_remain = size2 - (grid_z - 1) * z_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y, grid_z);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_broadcast_in2_dim02_grid3d<nt, vt, z_t, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, z_t, z_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
