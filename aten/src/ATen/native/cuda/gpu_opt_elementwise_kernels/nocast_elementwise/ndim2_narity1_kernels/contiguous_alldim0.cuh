// vt=1 path with P-element packing (P=4 scalar, P=8 vector-write).
// Each thread processes P elements along dim0 with scalar in1 reads;
// The P=8 variant upgrades to one aligned_vector<res_t, 8> store when out stride10 and
// data0 are 8*sizeof(res_t)-aligned (32B for float, 64B for double).
// vt >= 2 paths unchanged (vector read + vector write).
//
// The previous vt1_packed kernel kept a separate
// register array `arg0_t v[P]` whose tail slots (x_base + p >= size0) were
// never initialized; the guarded write loop then read them under extreme
// register pressure (C10_LAUNCH_BOUNDS_2(1024, 4)), which is UB and was
// miscompiled into deterministic cross-thread value corruption (A2 FAIL,
// Qwen3_vl-30b copy (N,3) transposed view). The kernel now fuses each
// element's load -> f -> store inside one guard, so no partially
// initialized array exists. The P==8 path additionally guards the 8-wide
// vector store with `x_base + 8 <= size0` and uses per-element scalar
// stores for the tail group; the old unconditional vector store wrote up
// to 7 elements out of bounds whenever size0 % 8 != 0.
template <int P, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_2_1_contiguous_alldim0_vt1_packed(
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f) {
  // ndim = 2, arity = 1. vt=1 packing variant: P scalar reads from in1,
  // P writes to out. When P==8, out uses one aligned_vector<res_t, 8> store.
  const int64_t x_base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * P;
  if (x_base >= size0) {
    return;
  }
  const int64_t y_b = static_cast<int64_t>(blockDim.y);
  const int64_t offset0_x = x_base * stride00;
  const int64_t offset1_x = x_base * stride01;

  for (int64_t y = static_cast<int64_t>(blockIdx.y) * y_b + threadIdx.y; y < size1;
       y += static_cast<int64_t>(gridDim.y) * y_b) {
    const int64_t offset0 = offset0_x + y * stride10;
    const int64_t offset1 = offset1_x + y * stride11;

    if constexpr (P == 8) {
      if (x_base + 8 <= size0) {
        // P=8 full group -> one aligned_vector<res_t, 8> store. The
        // launcher's k8b gate guarantees stride00 == sizeof(res_t) and both
        // stride10 and data0 aligned to 8 * sizeof(res_t), so the store
        // address is 8*sizeof(res_t)-aligned.
        using StoreT = at::native::memory::aligned_vector<res_t, 8>;
        StoreT st;
#pragma unroll
        for (int p = 0; p < 8; p++) {
          st.val[p] = f(*reinterpret_cast<const arg0_t*>(data1 + offset1 + p * stride01));
        }
        *reinterpret_cast<StoreT*>(data0 + offset0) = st;
      } else {
        // P=8 tail group (size0 % 8 != 0): per-element guarded scalar
        // stores; no partial vector store so nothing is written past the
        // row end.
#pragma unroll
        for (int p = 0; p < 8; p++) {
          if (x_base + p < size0) {
            *reinterpret_cast<res_t*>(data0 + offset0 + p * stride00) =
                f(*reinterpret_cast<const arg0_t*>(data1 + offset1 + p * stride01));
          }
        }
      }
    } else {
      // P=4 scalar reads and writes, fused per element. The tail group
      // (size0 % 4 != 0) is guarded per element and no register array with
      // uninitialized slots exists (see repair note above).
#pragma unroll
      for (int p = 0; p < P; p++) {
        if (x_base + p < size0) {
          *reinterpret_cast<res_t*>(data0 + offset0 + p * stride00) =
              f(*reinterpret_cast<const arg0_t*>(data1 + offset1 + p * stride01));
        }
      }
    }
  }
}


template <int vt, typename res_t, typename arg0_t, typename func_t>
C10_LAUNCH_BOUNDS_2(1024, 4)
__global__ void elementwise_kernel_2_1_contiguous_alldim0(
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    func_t f) {
  // ndim = 2, arity = 1. Both operands contiguous on dim0, arbitrary
  // (vector-aligned) dim1 strides: covers row-gap copies like [:, 0, :].
  // vt >= 2 only: vt==1 is handled by the _vt1_packed kernel above.

  // — blockIdx.x walks dim0 vectors (no per-thread 64-bit div/mod), the row
  // loop grid-strides over blockIdx.y with y_b == blockDim.y; gridDim.y is
  // capped at Y_CAP == 2048 by the launcher, so large size1 iterates the loop.
  // Dispatch guarantees vt == 1 whenever stride00 != sizeof(res_t), so the
  // vector store at offset0_x (x * stride00) is always correct.
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  const int64_t y_b = static_cast<int64_t>(blockDim.y);
  const int64_t offset0_x = x * stride00;
  const int64_t offset1_x = x * stride01;
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  for (int64_t y = static_cast<int64_t>(blockIdx.y) * y_b + threadIdx.y; y < size1;
       y += static_cast<int64_t>(gridDim.y) * y_b) {
    const int64_t offset0 = offset0_x + y * stride10;
    const int64_t offset1 = offset1_x + y * stride11;
    LoadT ld = *reinterpret_cast<const LoadT*>(data1 + offset1);
    StoreT st;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld.val[i]);
    }
    *reinterpret_cast<StoreT*>(data0 + offset0) = st;
  }
}


template <int vt, typename res_t, typename arg0_t, typename func_t>
static void launch_elementwise_kernel_2_1_contiguous_alldim0(
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
  constexpr int64_t y_cap = 2048;

  if constexpr (vt >= 2) {
    // Original vt >= 2 path: vector read + vector write.
    // Caller guarantees size0 % vt == 0.
    const int64_t x_vec = size0 / vt;
    const int64_t nt_x = std::min<int64_t>(x_vec, 128);
    const int64_t y_b = std::min<int64_t>(8, size1);
    dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
    const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
    const int64_t grid_y = std::min<int64_t>((size1 + y_b - 1) / y_b, y_cap);
    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
    auto stream = at::cuda::getCurrentCUDAStream();
    elementwise_kernel_2_1_contiguous_alldim0<vt, res_t, arg0_t, func_t>
        <<<grid, block, 0, stream>>>(
            data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // vt == 1: P=4 scalar read+write packing.
    // P=8: scalar read + one aligned_vector<res_t, 8> store. That type
    // requires 8 * sizeof(res_t) alignment (32B for float, 64B for double,
    // 16B for bf16), so out stride10 and the out data pointer must both be
    // 8*sizeof(res_t)-aligned. The vector store writes 8 *contiguous*
    // elements, so the P=8 variant additionally requires stride00 == sizeof(res_t);
    // strided dim0 (stride00 > es) stays on the P=4 scalar path, whose
    // stores honor p * stride00.
    const int64_t k8b_align = static_cast<int64_t>(sizeof(res_t)) * 8;
    const bool k8b = (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
                      stride10 % k8b_align == 0 &&
                      reinterpret_cast<uintptr_t>(data0) % k8b_align == 0);
    const int P = k8b ? 8 : 4;
    // Ceil: size0 % P != 0 must still launch a tail thread so the last
    // (size0 % P) elements of every row are written. The kernel guards
    // x_base >= size0 (early return) and x_base + p < size0 per element, so
    // the extra thread is safe.
    const int64_t x_vec = (size0 + P - 1) / P;
    const int64_t nt_x = std::min<int64_t>(x_vec, 128);
    const int64_t y_b = std::min<int64_t>(8, size1);
    dim3 block(static_cast<unsigned int>(nt_x), static_cast<unsigned int>(y_b));
    const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
    const int64_t grid_y = std::min<int64_t>((size1 + y_b - 1) / y_b, y_cap);
    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (k8b) {
      elementwise_kernel_2_1_contiguous_alldim0_vt1_packed<8, res_t, arg0_t, func_t>
          <<<grid, block, 0, stream>>>(
              data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    } else {
      elementwise_kernel_2_1_contiguous_alldim0_vt1_packed<4, res_t, arg0_t, func_t>
          <<<grid, block, 0, stream>>>(
              data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
