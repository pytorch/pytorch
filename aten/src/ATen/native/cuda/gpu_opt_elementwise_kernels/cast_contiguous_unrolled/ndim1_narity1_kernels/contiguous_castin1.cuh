// File: ndim1_narity1_kernels/contiguous_castin1.cuh
// tag: p_e_gpu_1_1_cast_v



// C1 patterns (cast_contiguous-unrolled, 1D fully contiguous arity-1 cast
// copy). vt_src/vt_dst are the per-side access widths (powers of two), each
// chosen by a down-scaling scan: the launcher walks (vt_src, vt_dst)
// combinations from the per-16B widths down to 1 and fires the first pair
// that divides numel and meets the pointer alignment. Each thread processes
// vt = max(vt_src, vt_dst) = lcm elements: the source side issues vt/vt_src
// aligned_vector<src_t> loads and the destination side vt/vt_dst
// aligned_vector<dst_t> stores, keeping both sides in-bounds and
// index-aligned. {Long,Bool} with numel % 4 == 0 reads 4 x 1B and writes
// 2 x 8B, {Float,Long} reads 2 x 8B and writes 4 x 4B, etc.
// tag p_e_gpu_1_1_cast_v.
//
// NOTE: no `namespace at::native` wrapper here. This file is included from
// dispatch.cuh inside CUDALoops.cuh's own `namespace at::native` block; a
// nested wrapper would introduce an `at` member shadowing every `at::`
// qualified reference (breaks CUDAJitLoops.cuh's `at::cuda::jit::` lookups).

#include "../../utils/common.cuh"

template <int vt_src, int vt_dst, int nt, typename dst_t, typename src_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void unrolled_elementwise_kernel_1_1_contiguous_castin1(
    int64_t N,
    char* data0, char* data1,
    func_t f) {
  constexpr int vt = vt_src > vt_dst ? vt_src : vt_dst;  // lcm (powers of two)
  constexpr int n_src = vt / vt_src;
  constexpr int n_dst = vt / vt_dst;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using vec_src = at::native::memory::aligned_vector<src_t, vt_src>;
  using vec_dst = at::native::memory::aligned_vector<dst_t, vt_dst>;
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  dst_t vals[vt];
#pragma unroll
  for (int j = 0; j < n_src; j++) {
    const vec_src in = *reinterpret_cast<const vec_src*>(
        data1 + (x + j * vt_src) * static_cast<int64_t>(sizeof(src_t)));
#pragma unroll
    for (int i = 0; i < vt_src; i++) {
      vals[j * vt_src + i] = f(c10::convert<f_arg0_t>(in.val[i]));
    }
  }
#pragma unroll
  for (int j = 0; j < n_dst; j++) {
    vec_dst out;
#pragma unroll
    for (int i = 0; i < vt_dst; i++) {
      out.val[i] = vals[j * vt_dst + i];
    }
    *reinterpret_cast<vec_dst*>(
        data0 + (x + j * vt_dst) * static_cast<int64_t>(sizeof(dst_t))) = out;
  }
}


// Down-scaling scan over (vt_src, vt_dst): walk the source width down to 1,
// and for each source width walk the destination width down to 1. The first
// combination that divides numel and meets both pointer alignments is
// launched. (1, 1) divides any numel and needs only natural alignment, so the
// scan always terminates with a launch when the pointers are naturally
// aligned; the caller's contiguous/dtype guards keep this family valid.
template <int vs, int vd, typename dst_t, typename src_t, typename func_t, typename array_t>
static inline bool try_launch_vt_scan(const func_t& f, array_t data, int64_t numel) {
  if (numel % vs == 0 && numel % vd == 0) {
    const int64_t al0 = static_cast<int64_t>(sizeof(dst_t)) * vd;
    const int64_t al1 = static_cast<int64_t>(sizeof(src_t)) * vs;
    if (reinterpret_cast<uintptr_t>(data[0]) % al0 == 0 &&
        reinterpret_cast<uintptr_t>(data[1]) % al1 == 0) {
      constexpr int nt = 256;
      constexpr int vt = vs > vd ? vs : vd;
      dim3 block(nt);
      dim3 grid((numel + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt));
      auto stream = at::cuda::getCurrentCUDAStream();
      unrolled_elementwise_kernel_1_1_contiguous_castin1<vs, vd, nt, dst_t, src_t, func_t>
          <<<grid, block, 0, stream>>>(numel, data[0], data[1], f);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return true;
    }
  }
  if constexpr (vd > 1) {
    return try_launch_vt_scan<vs, vd / 2, dst_t, src_t, func_t>(f, data, numel);
  } else if constexpr (vs > 1) {
    return try_launch_vt_scan<vs / 2, per_op_vector_width<dst_t>(), dst_t, src_t, func_t>(
        f, data, numel);
  } else {
    return false;
  }
}


template <typename dst_t, typename src_t, typename func_t, typename array_t>
static inline bool launch_unrolled_elementwise_kernel_1_1_contiguous_castin1(
    TensorIteratorBase& iter, const func_t& f, array_t data, int64_t numel) {
  // Compile-time guard: the {dst, src} dtype table may match functors
  // whose result type differs from dst_t (e.g. a complex mul wrapped as
  // arity-1). Such functors must not instantiate the cast kernel body.
  using f_res_t = typename function_traits<func_t>::result_type;
  if constexpr (!std::is_same_v<f_res_t, dst_t>) {
    return false;
  } else {
    return try_launch_vt_scan<per_op_vector_width<src_t>(), per_op_vector_width<dst_t>(),
                              dst_t, src_t, func_t>(f, data, numel);
  }
}
