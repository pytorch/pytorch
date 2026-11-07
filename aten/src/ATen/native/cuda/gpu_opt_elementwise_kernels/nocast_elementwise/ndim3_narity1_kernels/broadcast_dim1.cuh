
// IN is dim1-broadcast (stride11 == 0) and dim0-contiguous (stride01 == es):
// per thread, one vector load along dim0 feeds all size1 rows. grid.z walks
// dim2 in chunks of z_t (keeps gridDim.z within the 65535 hardware limit),
// grid.y walks rows in chunks of y_t.
// Mirror of elementwise_kernel_2_1_broadcast_dim1 with a dim2 batch axis

template <int nt, int vt, int z_t, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1_broadcast_dim1(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    func_t f, int64_t y_t, int64_t y_remain) {
  const int64_t z_base = static_cast<int64_t>(blockIdx.z) * z_t;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;

  // IN dim1-broadcast: one vector load on dim0 per (thread, z slice),
  // reused across all size1 rows.
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  const int64_t x_out = x * stride00;
  for (int z_i = 0; z_i < z_t && z_base + z_i < size2; z_i++) {
    const int64_t z = z_base + z_i;
    const LoadT ld = *reinterpret_cast<const LoadT*>(data1 + x * stride01 + z * stride21);
    StoreT st;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld.val[i]);
    }
    const int64_t z_out = z * stride20;
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      *reinterpret_cast<StoreT*>(data0 + z_out + y * stride10 + x_out) = st;
    }
  }
}

// When size0 is exactly one vector, only one x thread of the original 128-thread
// block is active. This path keeps unary-result reuse: the y==0 lane computes once
// per z, and the 8x8 y/z lanes only parallelize the independent output stores.
template <int vt, typename res_t, typename arg0_t, typename func_t>
__global__ void elementwise_kernel_3_1_broadcast_dim1_narrow_x(
    char* data0, char* data1, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride10,
    int64_t stride20, int64_t stride21, func_t f) {
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  __shared__ StoreT cache[8];
  const int64_t y = static_cast<int64_t>(blockIdx.y) * 8 + threadIdx.y;
  const int64_t z = static_cast<int64_t>(blockIdx.z) * 8 + threadIdx.z;
  if (threadIdx.y == 0 && z < size2) {
    const LoadT ld = *reinterpret_cast<const LoadT*>(data1 + z * stride21);
    StoreT st;
#pragma unroll
    for (int i = 0; i < vt; ++i) st.val[i] = f(ld.val[i]);
    cache[threadIdx.z] = st;
  }
  __syncthreads();
  if (y < size1 && z < size2) {
    *reinterpret_cast<StoreT*>(data0 + y * stride10 + z * stride20) =
        cache[threadIdx.z];
  }
}

template <int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_3_1_broadcast_dim1_narrow_x(
    char* data0, char* data1, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride10,
    int64_t stride20, int64_t stride21, const func_t& f) {
  dim3 block(1, 8, 8);
  dim3 grid(1, (size1 + 7) / 8, (size2 + 7) / 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_1_broadcast_dim1_narrow_x<vt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(data0, data1, size1, size2, stride00,
          stride01, stride10, stride20, stride21, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <int nt, int vt, int z_t, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_3_1_broadcast_dim1(
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
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y, (size2 + z_t - 1) / z_t);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_1_broadcast_dim1<nt, vt, z_t, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
