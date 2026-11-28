
// 3D reverse transpose: OUT contiguous on dim0, IN contiguous on dim1 (IN's
// (size1,size0) plane is the transpose view of OUT's (size0,size1) plane),
// dim2 is a batch axis walked by blockIdx.z. Tile read/write roles mirror

// offset. grid = (ceil(size1/tile), ceil(size0/tile), size2).
// tag p_e_gpu_3_1_tp.
template <int tile_dim, int block_rows, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(tile_dim * block_rows, 4)
__global__ void elementwise_kernel_3_1_transpose(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    int64_t stride20, int64_t stride21,
    func_t f) {
  // size0 = OUT dim0 (= IN dim1), size1 = OUT dim1 (= IN dim0).
  // Read: IN is dim1-contiguous — tx walks IN dim1 (size1), ty walks IN dim0
  // (size0).  tile[ty+i][tx] = in[(y+i)*stride01 + x*stride11 + n*stride21]
  // Write: OUT is dim0-contiguous — ox=r*tile+tx (OUT dim0), oy=c*tile+ty
  // (OUT dim1).  out[(oy+i)*stride10 + ox*stride00 + n*stride20] = tile[tx][ty+i]
  __shared__ res_t tile[tile_dim][tile_dim + 2];
  const int64_t c = blockIdx.x;  // IN dim1 tile (OUT dim1)
  const int64_t r = blockIdx.y;  // IN dim0 tile (OUT dim0)
  const int64_t n = blockIdx.z;  // dim2 batch
  const int64_t x = c * tile_dim + threadIdx.x;  // IN dim1 (= OUT dim1 coordinate)
  const int64_t y = r * tile_dim + threadIdx.y;  // IN dim0 (= OUT dim0 coordinate)
  if (x < size1) {
    for (int i = 0; threadIdx.y + i < tile_dim && y + i < size0; i += block_rows) {
      const int64_t offset1 = (y + i) * stride01 + x * stride11 + n * stride21;
      tile[threadIdx.y + i][threadIdx.x] =
          f(c10::load(reinterpret_cast<const arg0_t*>(data1 + offset1)));
    }
  }
  __syncthreads();
  const int64_t ox = r * tile_dim + threadIdx.x;  // OUT dim0
  const int64_t oy = c * tile_dim + threadIdx.y;  // OUT dim1
  if (ox < size0) {
    for (int i = 0; threadIdx.y + i < tile_dim && oy + i < size1; i += block_rows) {
      const int64_t offset0 = (oy + i) * stride10 + ox * stride00 + n * stride20;
      *reinterpret_cast<res_t*>(data0 + offset0) = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}


template <int tile_dim, int block_rows, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_3_1_transpose(
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
  dim3 block(tile_dim, block_rows);
  dim3 grid((size1 + tile_dim - 1) / tile_dim, (size0 + tile_dim - 1) / tile_dim, size2);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_1_transpose<tile_dim, block_rows, res_t, arg0_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, size0, size1, size2,
          stride00, stride01, stride10, stride11, stride20, stride21, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
