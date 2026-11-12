
// Input dim0 stride is a positive multiple of the element size (more than
// one element): every output element is gathered from a strided position in
// the input, so the input is loaded element-wise; the output is fully
// contiguous and vector-stored. Covers direct-copy layouts like
// (960,44064,2,2,3) whose input dim0 stride is 2 elements.
template <int nt, int vt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1_contiguous_out(
    int64_t N,
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    func_t f) {
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
  LoadT ld;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    ld.val[i] = c10::load(reinterpret_cast<const arg0_t*>(data1 + offset1 + i * stride01));
  }
  StoreT st;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    st.val[i] = f(ld.val[i]);
  }
  *reinterpret_cast<StoreT*>(data0 + offset0) = st;
}

// Shared-memory-free tile for permuted-contiguous-block (PCB) inputs. The wider
// dim1 and two dim2 lanes map directly to threads; the remaining high-dim
// coordinates come from the grid. This avoids the 5D linear-index decode of the
// generic strided path while keeping the original functor and dynamic strides.
template <typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(128, 4)
__global__ void elementwise_kernel_5_1_permuted_out_tile(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    int64_t dim2_tiles, func_t f) {
  constexpr int kDim1Tile = 64;
  constexpr int kDim2Tile = 2;
  constexpr int kVectorWidth = 4;
  const int64_t idx1 = static_cast<int64_t>(blockIdx.x) * kDim1Tile + threadIdx.x;
  const int64_t dim2_tile = static_cast<int64_t>(blockIdx.y) % dim2_tiles;
  const int64_t idx2 = dim2_tile * kDim2Tile + threadIdx.y;
  const int64_t idx3 = static_cast<int64_t>(blockIdx.y) / dim2_tiles;
  const int64_t idx4 = blockIdx.z;
  if (idx1 >= size1 || idx2 >= size2 || idx3 >= size3 || idx4 >= size4) {
    return;
  }
  const int64_t out_base = idx1 * stride10 + idx2 * stride20 +
      idx3 * stride30 + idx4 * stride40;
  const int64_t in_base = idx1 * stride11 + idx2 * stride21 +
      idx3 * stride31 + idx4 * stride41;
  using StoreT = at::native::memory::aligned_vector<res_t, kVectorWidth>;
  StoreT out;
#pragma unroll
  for (int i = 0; i < kVectorWidth; ++i) {
    out.val[i] = f(c10::load(reinterpret_cast<const arg0_t*>(
        data1 + in_base + i * stride01)));
  }
  *reinterpret_cast<StoreT*>(data0 + out_base) = out;
}

template <typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_5_1_permuted_out_tile(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    int64_t stride30, int64_t stride31,
    int64_t stride40, int64_t stride41,
    const func_t& f) {
  constexpr int kDim1Tile = 64;
  constexpr int kDim2Tile = 2;
  const int64_t dim2_tiles = (size2 + kDim2Tile - 1) / kDim2Tile;
  dim3 block(kDim1Tile, kDim2Tile);
  dim3 grid((size1 + kDim1Tile - 1) / kDim1Tile, dim2_tiles * size3, size4);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1_permuted_out_tile<res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, size0, size1, size2, size3, size4,
          stride00, stride01, stride10, stride11, stride20, stride21,
          stride30, stride31, stride40, stride41, dim2_tiles, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_5_1_contiguous_out(
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
  elementwise_kernel_5_1_contiguous_out<nt, vt, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, size3, size4,
          stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
