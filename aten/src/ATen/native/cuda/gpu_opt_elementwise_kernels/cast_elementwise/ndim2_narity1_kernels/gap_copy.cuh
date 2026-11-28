
// C7 patterns ((1152,23040), (1152,7040), (1152,23712)): OUT float
// dim0-contiguous, IN bf16 dim0-contiguous; both row strides may carry gaps
// (IN rows are slices of a larger tensor). Each thread handles 8 elements
// (16B of IN, 32B of OUT written as two 16B stores). tag p_e_gpu_2_1_cast_gap.
template <int nt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void cast_elementwise_kernel_2_1_gap_copy(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride10, int64_t stride11,
    func_t f, int64_t y_t, int64_t y_remain) {
  constexpr int vt = 8;  // lcm(4 float, 8 bf16) elements per thread
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;
  using vec_in = at::native::memory::aligned_vector<c10::BFloat16, 8>;
  using vec_out = at::native::memory::aligned_vector<float, 4>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
    const int64_t y = y_start + y_idx;
    const vec_in in = *reinterpret_cast<const vec_in*>(data1 + y * stride11 + x * stride01);
    float vals[vt];
#pragma unroll
    for (int i = 0; i < vt; i++) {
      vals[i] = f(c10::convert<f_arg0_t>(in.val[i]));
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      vec_out out;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        out.val[i] = vals[j * 4 + i];
      }
      *reinterpret_cast<vec_out*>(data0 + y * stride10 + (x + j * 4) * stride00) = out;
    }
  }
}


template <int nt, typename func_t>
static void launch_cast_elementwise_kernel_2_1_gap_copy(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride10, int64_t stride11,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * 8 - 1) / (static_cast<int64_t>(nt) * 8);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y);
  auto stream = at::cuda::getCurrentCUDAStream();
  cast_elementwise_kernel_2_1_gap_copy<nt, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
