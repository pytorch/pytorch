
// Reverse transpose: OUT is contiguous on dim0, IN on dim1: the logical
// transpose direction is reversed (IN is the transposed view of OUT, not
// OUT of IN).  grid = (ceil(size1/tile_dim), ceil(size0/tile_dim));

//   out[(oy+i)*stride10 + ox*stride00] = tile[tx][ty+i]
// with IN read along the contiguous dim1 and OUT written along dim0.
template <int tile_dim, int block_rows, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(tile_dim * block_rows, 4)
__global__ void elementwise_kernel_2_1_transpose_reverse(
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f) {
  // size0 = W (out dim0), size1 = H (out dim1).
  // grid = (ceil(size1/tile_dim), ceil(size0/tile_dim)).
  //
  // Read: IN is dim1-contiguous — tx walks dim1 (size1), ty walks dim0 (size0).
  //   x = c*tile + tx  (IN dim1), y = r*tile + ty  (IN dim0)
  //   tile[ty+i][tx] = in[(y+i)*stride01 + x*stride11]
  // Write: OUT is dim0-contiguous — ox=r*tile+tx (OUT dim0), oy=c*tile+ty (OUT dim1).
  //   out[(oy+i)*stride10 + ox*stride00] = tile[tx][ty+i]
  __shared__ res_t tile[tile_dim][tile_dim + 2];
  const int64_t c = blockIdx.x;  // col in size1 grid (IN dim1 tile)
  const int64_t r = blockIdx.y;  // row in size0 grid (IN dim0 tile)
  const int64_t x = c * tile_dim + threadIdx.x;  // IN dim1 (= OUT dim1 coordinate)
  const int64_t y = r * tile_dim + threadIdx.y;  // IN dim0 (= OUT dim0 coordinate)
  if (x < size1) {
    for (int i = 0; threadIdx.y + i < tile_dim && y + i < size0; i += block_rows) {
      const int64_t offset1 = (y + i) * stride01 + x * stride11;
      tile[threadIdx.y + i][threadIdx.x] =
          f(c10::load(reinterpret_cast<const arg0_t*>(data1 + offset1)));
    }
  }
  __syncthreads();
  const int64_t ox = r * tile_dim + threadIdx.x;  // OUT dim0
  const int64_t oy = c * tile_dim + threadIdx.y;  // OUT dim1
  if (ox < size0) {
    for (int i = 0; threadIdx.y + i < tile_dim && oy + i < size1; i += block_rows) {
      const int64_t offset0 = (oy + i) * stride10 + ox * stride00;
      *reinterpret_cast<res_t*>(data0 + offset0) = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}


template <int tile_dim, int block_rows, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_2_1_transpose_reverse(
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
  dim3 block(tile_dim, block_rows);
  dim3 grid((size1 + tile_dim - 1) / tile_dim, (size0 + tile_dim - 1) / tile_dim);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1_transpose_reverse<tile_dim, block_rows, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
