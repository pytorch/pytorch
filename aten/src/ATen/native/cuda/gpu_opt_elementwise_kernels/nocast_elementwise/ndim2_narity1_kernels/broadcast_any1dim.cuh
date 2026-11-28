

// Policy functor encapsulates the shape asymmetry between dim0-broadcast
// (strip_bd0: scalar row read + splat store) and dim1-broadcast
// (strip_bd1: vector read hoisted out of the row loop + cross-row reuse
// store); the kernel body is a zero-if/else common skeleton. One policy is
// instantiated per kernel, equivalent to the former two standalone kernels.
// tag p_e_gpu_2_1_1db.

// Policy 1: IN broadcast on dim0 -- one scalar load per row, f once per
// row, result splatted into the vt-wide vector store.
template <int vt, typename res_t, typename arg0_t, typename func_t>
struct strip_bd0 {
  static constexpr bool arg_needs_align = false;  // IN scalar read, no vt align
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  __device__ static void run(
      char* data0, char* data1,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t, int64_t stride10, int64_t stride11,
      const func_t& f) {
    #pragma unroll 2
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      const arg0_t in1 = c10::load(reinterpret_cast<const arg0_t*>(data1 + y * stride11));
      const res_t val = f(in1);
      StoreT st;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        st.val[i] = val;
      }
      *reinterpret_cast<StoreT*>(data0 + y * stride10 + x * stride00) = st;
    }
  }
};

// Policy 2: IN broadcast on dim1 -- vt-wide vector read once, hoisted out of
// the row loop, f per lane, the same vector reused across rows.
template <int vt, typename res_t, typename arg0_t, typename func_t>
struct strip_bd1 {
  static constexpr bool arg_needs_align = true;   // IN vector read, needs vt align
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  __device__ static void run(
      char* data0, char* data1,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t stride01, int64_t stride10, int64_t,
      const func_t& f) {
    const LoadT ld = *reinterpret_cast<const LoadT*>(data1 + x * stride01);
    StoreT st;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld.val[i]);
    }
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      *reinterpret_cast<StoreT*>(data0 + y * stride10 + x * stride00) = st;
    }
  }
};

// Common skeleton: zero if/else, a single Policy::run call at the end.
template <typename Policy, int nt, int vt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_broadcast_any1dim(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f, int64_t y_t, int64_t y_remain) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;

  Policy::run(data0, data1, x, y_start, y_loop,
              stride00, stride01, stride10, stride11, f);
}


template <typename Policy, int nt, int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_2_1_broadcast_any1dim(
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
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1_broadcast_any1dim<Policy, nt, vt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
