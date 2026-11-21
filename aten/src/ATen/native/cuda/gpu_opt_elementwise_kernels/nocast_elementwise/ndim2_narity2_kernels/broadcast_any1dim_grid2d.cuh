

// Policy functor encapsulates the shape asymmetry between dim0-broadcast
// (strip_d0a1: IN2 per-row scalar read) and dim1-broadcast
// (strip_d1a1 / strip_d1a0: hoisted vector read + cross-row reuse); the
// kernel body is a zero-if/else common skeleton. Traits compile-time the
// dispatch differences (row-read operand, IN2 vector alignment, block size,
// 1D-grid fallback). tag p_e_gpu_2_2_1db_2d / p_e_gpu_2_2_1db_1d.

// Policy 1: arg1 dim0-broadcast (a1d0) -- IN2 per-row scalar read, IN1
// vector read per row. nt=256 halves the grid size vs 128 for the small
// launch-bound add/mul workloads; carries the 1D-grid fallback for
// size1 > 65535 * y_t.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
struct strip_d0a1 {
  static constexpr bool row_stride_is_s11 = true;   // row-read operand is IN1 (stride11)
  static constexpr bool data2_vec_read = false;     // IN2 scalar read, no vector align check
  static constexpr int nt = 256;
  static constexpr bool has_1d_fallback = true;
  static constexpr const char* tag_2d = "p_e_gpu_2_2_1db_2d";
  static constexpr const char* tag_1d = "p_e_gpu_2_2_1db_1d";
  __device__ static void run(
      char* data0, char* data1, char* data2,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t stride01, int64_t, int64_t stride10, int64_t stride11, int64_t stride12,
      const func_t& f) {
    using vec_res = at::native::memory::aligned_vector<res_t, vt>;
    using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
    // B1 row-loop unroll knob: cap the compiler's partial unroll at 2 so
    // heavy ops keep per-thread work small (occupancy), light ops keep a
    // 2-row overlap window for latency hiding.
    #pragma unroll 2
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + x * stride01);
      const arg1_t in2 = c10::load(reinterpret_cast<const arg1_t*>(data2 + y * stride12));
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(in1.val[i], in2);
      }
      *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
    }
  }
};

// Policy 2: arg1 dim1-broadcast (b_a1d1) -- IN2 vector read hoisted out of
// the row loop and reused across rows. Also covers arg1 == Bool (1 byte):
// masked_fill with a Bool mask broadcast on dim1 (P2).
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
struct strip_d1a1 {
  static constexpr bool row_stride_is_s11 = true;
  static constexpr bool data2_vec_read = true;      // IN2 vector read (Bool at its own granularity)
  static constexpr int nt = 128;
  static constexpr bool has_1d_fallback = false;
  static constexpr const char* tag_2d = "p_e_gpu_2_2_1db_2d";
  static constexpr const char* tag_1d = "p_e_gpu_2_2_1db_1d";
  __device__ static void run(
      char* data0, char* data1, char* data2,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t stride01, int64_t stride02, int64_t stride10, int64_t stride11, int64_t,
      const func_t& f) {
    using vec_res = at::native::memory::aligned_vector<res_t, vt>;
    using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
    using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
    const vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + x * stride02);
    #pragma unroll 2
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + x * stride01);
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(in1.val[i], in2.val[i]);
      }
      *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
    }
  }
};

// Policy 3: arg0 dim1-broadcast (b_a0d1) -- IN1 vector read hoisted out of
// the row loop, IN2 vector read per row. The former swap kernel's role
// (keeping f's argument order) dissolves: the policy body itself fixes the
// f(arg0, arg1) call order, unlike swapping the data pointers.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
struct strip_d1a0 {
  static constexpr bool row_stride_is_s11 = false;  // row-read operand is IN2 (stride12)
  static constexpr bool data2_vec_read = true;
  static constexpr int nt = 128;
  static constexpr bool has_1d_fallback = false;
  static constexpr const char* tag_2d = "p_e_gpu_2_2_1db_2d";
  static constexpr const char* tag_1d = "p_e_gpu_2_2_1db_1d";
  __device__ static void run(
      char* data0, char* data1, char* data2,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t stride01, int64_t stride02, int64_t stride10, int64_t, int64_t stride12,
      const func_t& f) {
    using vec_res = at::native::memory::aligned_vector<res_t, vt>;
    using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
    using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
    const vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + x * stride01);
    #pragma unroll 2
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + y * stride12 + x * stride02);
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(in1.val[i], in2.val[i]);
      }
      *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
    }
  }
};

// Policy 4: arg0 dim0-broadcast (a0d0) -- IN1 per-row scalar read, IN2
// vector read per row. Mirror of strip_d0a1 with the broadcast operand on
// the arg0 side; IN1 is read once per row and reused across the vt
// elements of that row. Same-width arg1 only; no 1D-grid fallback.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
struct strip_d0a0 {
  static constexpr bool row_stride_is_s11 = false;  // row-read operand is IN2 (stride12)
  static constexpr bool data2_vec_read = true;      // IN2 vector read per row
  static constexpr int nt = 256;
  static constexpr bool has_1d_fallback = false;
  static constexpr const char* tag_2d = "p_e_gpu_2_2_d0a0";
  static constexpr const char* tag_1d = "p_e_gpu_2_2_d0a0";
  __device__ static void run(
      char* data0, char* data1, char* data2,
      int64_t x, int64_t y_start, int64_t y_loop,
      int64_t stride00, int64_t, int64_t stride02, int64_t stride10, int64_t stride11, int64_t stride12,
      const func_t& f) {
    using vec_res = at::native::memory::aligned_vector<res_t, vt>;
    using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
    #pragma unroll 2
    for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
      const int64_t y = y_start + y_idx;
      // IN1 dim0-broadcast: one scalar per row, independent of x.
      const arg0_t in1 = c10::load(reinterpret_cast<const arg0_t*>(data1 + y * stride11));
      vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + y * stride12 + x * stride02);
      vec_res out;
#pragma unroll
      for (int i = 0; i < vt; i++) {
        out.val[i] = f(in1, in2.val[i]);
      }
      *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
    }
  }
};

// Common skeleton: zero if/else, a single Policy::run call at the end.
// Reuses the b_a1d1 kernel's established name.
template <typename Policy, int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_broadcast_any1dim_grid2d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
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

  Policy::run(data0, data1, data2, x, y_start, y_loop,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
}

// Narrow-x variant of strip_d1a1: at size0=64, vt=8 only 8 x lanes of the old
// 128-thread block are active. An 8x16 block parallelizes 16 output rows and keeps
// the original dim1-broadcast input reuse in shared memory.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
__global__ void elementwise_kernel_2_2_broadcast_d1a1_narrow_x(
    char* data0, char* data1, char* data2, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, func_t f) {
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  __shared__ arg1_t in2[64];
  const int x_lane = threadIdx.x;
  if (threadIdx.y == 0) {
#pragma unroll
    for (int i = 0; i < vt; ++i)
      in2[x_lane * vt + i] = *reinterpret_cast<const arg1_t*>(data2 + (x_lane * vt + i) * stride02);
  }
  __syncthreads();
  const int64_t y = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.y;
  if (y >= size1) return;
  const int64_t x = static_cast<int64_t>(x_lane) * vt;
  const vec_arg0 a = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + x * stride01);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; ++i) out.val[i] = f(a.val[i], in2[x + i]);
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
}


template <typename Policy, int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  if constexpr (std::is_same_v<Policy, strip_d1a1<vt, res_t, arg0_t, arg1_t, func_t>>) {
    if (size0 == 64 && vt == 8) {
      dim3 block(8, 16);
      dim3 grid(1, (size1 + 15) / 16);
      auto stream = at::cuda::getCurrentCUDAStream();
      elementwise_kernel_2_2_broadcast_d1a1_narrow_x<vt, res_t, arg0_t, arg1_t, func_t>
          <<<grid, block, 0, stream>>>(data0, data1, data2, size1,
              stride00, stride01, stride02, stride10, stride11, f);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_2_broadcast_any1dim_grid2d<Policy, nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
