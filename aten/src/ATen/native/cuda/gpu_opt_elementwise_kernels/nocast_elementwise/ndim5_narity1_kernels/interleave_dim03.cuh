
// IN (d0, d3)-interleaved, OUT fully contiguous (mirror direction of
// elementwise_kernel_5_1_interleave_d0_d3). IN's (d0, d3) plane is one
// contiguous vt*s3-element run (IN dim0 strides by size3 elements and dim3
// by one element), so it is read with a single vt*s3-wide vector load; OUT
// receives s3 separate vt-wide vector stores along its contiguous dim0
// (OUT dim3 strides by size0 elements). No data pointer swap: data0 is the
// contiguous OUT, data1 is the interleaved IN, matching the functor
// direction res = f(arg0).
template <int nt, int vt, int s3, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1_interleave_dim03(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    func_t f) {
  const int64_t rows_per_d0_block = size0 / vt;
  const int64_t seg_linear = static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x;
  const int64_t total_segments = rows_per_d0_block * size1 * size2 * size4;
  if (seg_linear >= total_segments) {
    return;
  }
  const int64_t d0_seg = seg_linear % rows_per_d0_block;
  const int64_t idx1 = (seg_linear / rows_per_d0_block) % size1;
  const int64_t idx2 = (seg_linear / (rows_per_d0_block * size1)) % size2;
  const int64_t idx4 = seg_linear / (rows_per_d0_block * size1 * size2);
  const int64_t d0_base = d0_seg * vt;

  const int64_t out_base = idx1 * stride10 + idx2 * stride20 + idx4 * stride40 + d0_base * stride00;
  const int64_t in_base = idx1 * stride11 + idx2 * stride21 + idx4 * stride41 + d0_base * stride01;

  using LoadT = at::native::memory::aligned_vector<arg0_t, vt * s3>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  LoadT ld = *reinterpret_cast<const LoadT*>(data1 + in_base);
  StoreT st;
#pragma unroll
  for (int k = 0; k < s3; k++) {
#pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld.val[i * s3 + k]);
    }
    *reinterpret_cast<StoreT*>(data0 + out_base + k * stride30) = st;
  }
}


template <int nt, int vt, int s3, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_5_1_interleave_dim03(
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
  TORCH_INTERNAL_ASSERT(size3 == s3);
  if (N == 0) {
    return;
  }
  const int64_t total_segments = (size0 / vt) * size1 * size2 * size4;
  dim3 block(nt);
  dim3 grid((total_segments + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1_interleave_dim03<nt, vt, s3, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, size4,
          stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
