
// IN2 is dim1-broadcast (stride12 == 0) and dim0-contiguous; OUT and IN1 are
// fully contiguous. Only IN2 is invariant along dim1, so it is read once per
// z segment and reused across the y_t rows; IN1 varies along dim1
// (stride11 == es * size0) and is therefore loaded, computed and stored once
// per row. IN2 stays dim0-contiguous, so all vt elements along dim0 differ:
// they come from one aligned vector load when sizeof(arg1_t) == es, and are
// read element-wise otherwise (e.g. a Bool mask). grid.z walks dim2 in z_t
// chunks (tag p_e_gpu_3_2_bd1).
template <int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_3_2_broadcast_in2_dim1(
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    func_t f) {
  // IN2 is dim1-broadcast (stride12 == 0) and dim0-contiguous; OUT and IN1 are

  // elementwise_kernel_3_2_broadcast_dim1 — blockIdx.x walks dim0 vectors
  // (blockDim.x == nt_x <= 128, no per-thread 64-bit div/mod), blockIdx.y walks
  // dim1 rows in chunks of y_b == blockDim.y, blockIdx.z walks dim2 in z_t
  // chunks. z rounds are distributed across threadIdx.y lanes: each (x vector,
  // z segment) is owned by exactly one lane, which loads IN2 once and reuses it
  // across all y_b rows of the block (IN2 read count unchanged from the old
  // kernel: once per (x, z segment)). ub == 4 batches the per-z loads for
  // es == 16 (vt == 1) exactly like the old kernel; every other dtype keeps
  // ub == 1 and the loop degenerates to the original per-z code (constant
  // folding, zero overhead).
  constexpr int ub = (sizeof(res_t) == 16) ? 4 : 1;
  const int64_t z_base = static_cast<int64_t>(blockIdx.z) * z_t;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y_b = static_cast<int64_t>(blockDim.y);
  const int64_t y_base = static_cast<int64_t>(blockIdx.y) * y_b;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  const int64_t x_out = x * stride00;
  const char* in1_base = data1 + x * stride01;
  const char* in2_base = data2 + x * stride02;
  for (int z_i = static_cast<int>(threadIdx.y) * ub; z_i < z_t;
       z_i += static_cast<int>(blockDim.y) * ub) {
    const int64_t z0 = z_base + z_i;
    if (z0 >= size2) {
      break;
    }
    // IN2 is the only dim1-broadcast operand: read it once per z segment and
    // reuse it across every y_b row. It is dim0-contiguous, so the vt elements
    // along dim0 all differ and each one has to be loaded.
    arg1_t in2[ub][vt];
#pragma unroll
    for (int uu = 0; uu < ub; uu++) {
      if (z0 + uu < size2) {
        const char* p2 = in2_base + (z0 + uu) * stride22;
        if constexpr (sizeof(arg1_t) == sizeof(res_t)) {
          // Matching element size: one aligned vector load (the caller has
          // checked es * vt alignment) covers the vt dim0 elements.
          using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
          const vec_arg1 v = *reinterpret_cast<const vec_arg1*>(p2);
#pragma unroll
          for (int i = 0; i < vt; i++) {
            in2[uu][i] = v.val[i];
          }
        } else {
          // Narrower operand (e.g. a Bool mask): only natural alignment is
          // guaranteed, so the vt dim0 elements are read one by one.
#pragma unroll
          for (int i = 0; i < vt; i++) {
            in2[uu][i] = c10::load(reinterpret_cast<const arg1_t*>(
                p2 + i * static_cast<int64_t>(sizeof(arg1_t))));
          }
        }
      }
    }
    // IN1 and OUT are fully contiguous, so both advance along dim1: load,
    // compute and store per y row. Hoisting the IN1 load out of this loop
    // would reuse row y_base for every row (stride11 != 0 here).
    const char* in1_z = in1_base + z0 * stride21;
    char* out_z = data0 + x_out + z0 * stride20;
    #pragma unroll 2
    for (int64_t ty = 0; ty < y_b; ty++) {
      const int64_t yy = y_base + ty;
      if (yy >= size1) {
        break;
      }
      const char* row_in1 = in1_z + yy * stride11;
      char* row_out = out_z + yy * stride10;
      // load phase: issue all ub loads back-to-back so their latencies overlap
      vec_arg0 in1[ub];
#pragma unroll
      for (int uu = 0; uu < ub; uu++) {
        if (z0 + uu < size2) {
          in1[uu] = *reinterpret_cast<const vec_arg0*>(row_in1 + uu * stride21);
        }
      }
      // compute + store phase
#pragma unroll
      for (int uu = 0; uu < ub; uu++) {
        if (z0 + uu < size2) {
          vec_res out;
#pragma unroll
          for (int i = 0; i < vt; i++) {
            out.val[i] = f(in1[uu].val[i], in2[uu][i]);
          }
          *reinterpret_cast<vec_res*>(row_out + uu * stride20) = out;
        }
      }
    }
  }
}

// Narrow-x variant for size0=16, vt=8: unroll the original serial y loop and z
// segment into block.y/block.z; shared IN2 preserves dim1-broadcast reuse across y.
template <int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
__global__ void elementwise_kernel_3_2_broadcast_in2_dim1_narrow_x(
    char* data0, char* data1, char* data2, int64_t size1, int64_t size2,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride20, int64_t stride21,
    int64_t stride22, func_t f) {
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  __shared__ arg1_t in2[8][16];
  const int x_lane = threadIdx.x, z_lane = threadIdx.z;
  const int64_t z = static_cast<int64_t>(blockIdx.z) * 8 + z_lane;
  if (threadIdx.y == 0 && z < size2) {
#pragma unroll
    for (int i = 0; i < vt; ++i)
      in2[z_lane][x_lane * vt + i] = *reinterpret_cast<const arg1_t*>(data2 + z * stride22 + (x_lane * vt + i) * stride02);
  }
  __syncthreads();
  const int64_t y = static_cast<int64_t>(blockIdx.y) * 8 + threadIdx.y;
  if (y >= size1 || z >= size2) return;
  const int64_t x = static_cast<int64_t>(x_lane) * vt;
  const vec_arg0 a = *reinterpret_cast<const vec_arg0*>(data1 + y * stride11 + z * stride21 + x * stride01);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; ++i) out.val[i] = f(a.val[i], in2[z_lane][x + i]);
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + z * stride20 + x * stride00) = out;
}


template <int vt, int z_t, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_3_2_broadcast_in2_dim1(
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
  if (size0 == 16 && (vt == 8 || vt == 4)) {
    dim3 block(static_cast<unsigned int>(16 / vt), 8, 8);
    dim3 grid(1, (size1 + 7) / 8, (size2 + 7) / 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    elementwise_kernel_3_2_broadcast_in2_dim1_narrow_x<vt, res_t, arg0_t, arg1_t, func_t>
        <<<grid, block, 0, stream>>>(data0, data1, data2, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride20,
            stride21, stride22, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Caller guarantees size0 % vt == 0 and size1 <= 65535 * 8 (gridDim.y limit).
  const int64_t x_vec = size0 / vt;
  const int64_t nt_x = std::min<int64_t>(x_vec, 128);
  const int64_t y_b = std::min<int64_t>(8, size1);
  dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
  const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
  const int64_t grid_y = (size1 + y_b - 1) / y_b;
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>((size2 + z_t - 1) / z_t));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_broadcast_in2_dim1<vt, z_t, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          data0, data1, data2, size0, size1, size2,
          stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
