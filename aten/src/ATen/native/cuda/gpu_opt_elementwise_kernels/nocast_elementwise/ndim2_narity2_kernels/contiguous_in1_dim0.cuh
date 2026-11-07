// Shmem tile variant for in2_lane_scalar_read + vt=8.
// Block covers nt_x*vt dim0 elements × y_b dim1 rows; cooperative load of
// in2 along dim1 (contiguous, stride12==es) into shmem (~2KB/block), then
// out/in1 vector read along dim0 + in2 from shmem.  vt <= 4 and in2_vec_read
// paths unchanged.

// reader functor encapsulates the IN2 read shape (vector load vs per-lane
// scalar load); the kernel body is a zero-if/else common skeleton
// `st.val[i] = f(ld0.val[i], r2[i])`, the reader's operator[] inlines to
// the former per-lane load. Both forms are 1D per-vector kernels (one
// div/mod per thread, no y loop), so there is no gridDim.y ceiling. tag
// p_e_gpu_2_2_1dc.

// Reader 1: IN2 vector read (d0c: dim0-contiguous).
struct in2_vec_read {
  static constexpr bool vec_read = true;
  template <int vt, typename arg1_t>
  struct reader {
    const at::native::memory::aligned_vector<arg1_t, vt> ld1;
    __device__ reader(const char* data2, int64_t offset2, int64_t /*stride02*/)
        : ld1(*reinterpret_cast<const at::native::memory::aligned_vector<arg1_t, vt>*>(
              data2 + offset2)) {}
    __device__ arg1_t operator[](int i) const { return ld1.val[i]; }
  };
};

// Reader 2: IN2 per-lane scalar load (d0c-reorder: dim1-contiguous).
struct in2_lane_scalar_read {
  static constexpr bool vec_read = false;
  template <int vt, typename arg1_t>
  struct reader {
    const char* data2;
    int64_t offset2;
    int64_t stride02;
    __device__ reader(const char* d2, int64_t o2, int64_t s02)
        : data2(d2), offset2(o2), stride02(s02) {}
    __device__ arg1_t operator[](int i) const {
      return *reinterpret_cast<const arg1_t*>(data2 + offset2 + i * stride02);
    }
  };
};

// Shmem tile kernel for in2_lane_scalar_read + vt=8.
// nt_x × vt dim0 elements, y_b dim1 rows per block; in2 tile loaded
// cooperatively along dim1 (contiguous) into shmem, then out/in1 vector
// read along dim0 and in2 from shmem.
template <int nt_x, int vt, int y_b,
          typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt_x, 4)
__global__ void elementwise_kernel_2_2_contiguous_in1_dim0_shmem_tile(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    func_t f) {
  // Block covers: dim0 [bx*nt_x*vt, (bx+1)*nt_x*vt) × dim1 [by*y_b, (by+1)*y_b).
  // Each thread processes vt elements along dim0 within one row.
  constexpr int tile_dim0 = nt_x * vt;  // dim0 elements per block
  __shared__ arg1_t in2_tile[tile_dim0][y_b];  // [dim0_idx][dim1_row]

  const int64_t tile_dim0_start = static_cast<int64_t>(blockIdx.x) * tile_dim0;
  const int64_t tile_dim1_start = static_cast<int64_t>(blockIdx.y) * y_b;

  // Cooperative load of in2 tile: each thread loads multiple (d0,d1) pairs,
  // striding along dim0 by nt_x.  The innermost loop loads along dim1
  // (contiguous, stride12==es) for full transactions.
  for (int d0 = threadIdx.x; d0 < tile_dim0; d0 += nt_x) {
    int64_t global_d0 = tile_dim0_start + d0;
    if (global_d0 >= size0) continue;
    int64_t off2_base = global_d0 * stride02 + tile_dim1_start * stride12;
    for (int d1 = 0; d1 < y_b; d1++) {
      int64_t global_d1 = tile_dim1_start + d1;
      if (global_d1 < size1) {
        in2_tile[d0][d1] = *reinterpret_cast<const arg1_t*>(data2 + off2_base + d1 * stride12);
      }
    }
  }
  __syncthreads();

  // Compute: each thread handles vt elements along dim0, iterating rows.
  int64_t idx0 = tile_dim0_start + static_cast<int64_t>(threadIdx.x) * vt;
  if (idx0 >= size0) {
    return;
  }
  const int64_t off0_x = idx0 * stride00;
  const int64_t off1_x = idx0 * stride01;

  using LoadT0 = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;

  for (int d1 = 0; d1 < y_b; d1++) {
    int64_t idx1 = tile_dim1_start + d1;
    if (idx1 >= size1) break;

    const int64_t off0 = off0_x + idx1 * stride10;
    const int64_t off1 = off1_x + idx1 * stride11;

    LoadT0 ld0 = *reinterpret_cast<const LoadT0*>(data1 + off1);
    StoreT st;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld0.val[i], in2_tile[threadIdx.x * vt + i][d1]);
    }
    *reinterpret_cast<StoreT*>(data0 + off0) = st;
  }
}

// Common skeleton: zero if/else, a single In2Read reader construction and
// one f call per lane.
template <typename In2Read, int nt, int vt,
          typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_contiguous_in1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    func_t f) {
  int64_t linear_idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (linear_idx >= N) {
    return;
  }
  const int64_t idx0 = linear_idx % size0;
  const int64_t idx1 = linear_idx / size0;
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11;
  const int64_t offset2 = idx0 * stride02 + idx1 * stride12;

  using LoadT0 = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  LoadT0 ld0 = *reinterpret_cast<const LoadT0*>(data1 + offset1);
  typename In2Read::template reader<vt, arg1_t> r2(data2, offset2, stride02);
  StoreT st;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    st.val[i] = f(ld0.val[i], r2[i]);
  }
  *reinterpret_cast<StoreT*>(data0 + offset0) = st;
}


template <typename In2Read, int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_2_2_contiguous_in1_dim0(
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

  // Shmem tile variant for in2_lane_scalar_read + vt=8.
  // Block covers tile_dim0 × y_b elements; grid is 2D.
  if constexpr (!In2Read::vec_read && vt == 8) {
    constexpr int nt_x_k9 = 16;
    constexpr int y_b = 8;
    constexpr int tile_dim0 = nt_x_k9 * vt;  // 128
    const int64_t grid_x = (size0 + tile_dim0 - 1) / tile_dim0;
    const int64_t grid_y = (size1 + y_b - 1) / y_b;
    dim3 block(nt_x_k9);
    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
    auto stream = at::cuda::getCurrentCUDAStream();
    elementwise_kernel_2_2_contiguous_in1_dim0_shmem_tile<nt_x_k9, vt, y_b, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(
            N, data0, data1, data2, size0, size1,
            stride00, stride01, stride02, stride10, stride11, stride12, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  // Original 1D grid path (vt <= 4, or in2_vec_read).
  dim3 block(nt);
  dim3 grid((N + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_2_contiguous_in1_dim0<In2Read, nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}