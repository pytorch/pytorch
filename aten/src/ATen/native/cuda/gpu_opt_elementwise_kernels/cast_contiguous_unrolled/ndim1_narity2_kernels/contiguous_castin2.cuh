// File: ndim1_narity2_kernels/contiguous_castin2.cuh
// tag: p_e_gpu_1_2_cast_v



// C2 patterns (cast_contiguous-unrolled, 1D fully contiguous arity-2 cast
// add/mul with a BFloat16/Long/Int second operand cast on load). Same

// vt = max(16/sizeof(res), 16/sizeof(in1), 16/sizeof(in2)) elements, loading
// both inputs in 16B chunks and storing the output in 16B chunks.
// tag p_e_gpu_1_2_cast_v.
//
// NOTE: no `namespace at::native` wrapper here. This file is included from
// dispatch.cuh inside CUDALoops.cuh's own `namespace at::native` block; a
// nested wrapper would introduce an `at` member shadowing every `at::`
// qualified reference (breaks CUDAJitLoops.cuh's `at::cuda::jit::` lookups).

#include "../../utils/common.cuh"

template <int vt0, int vt1, int vt2, int nt, typename res_t, typename in1_t, typename in2_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void unrolled_elementwise_kernel_1_2_contiguous_castin2(
    int64_t N,
    char* data0, char* data1, char* data2,
    func_t f) {
  constexpr int vt = (vt0 > vt1 ? (vt0 > vt2 ? vt0 : vt2) : (vt1 > vt2 ? vt1 : vt2));
  constexpr int n0 = vt / vt0;
  constexpr int n1 = vt / vt1;
  constexpr int n2 = vt / vt2;
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= N) {
    return;
  }
  using f_traits = function_traits<func_t>;
  using f_arg0_t = typename f_traits::template arg<0>::type;
  using f_arg1_t = typename f_traits::template arg<1>::type;
  using vec_in1 = at::native::memory::aligned_vector<in1_t, vt1>;
  using vec_in2 = at::native::memory::aligned_vector<in2_t, vt2>;
  using vec_res = at::native::memory::aligned_vector<res_t, vt0>;
  f_arg0_t v1[vt];
  f_arg1_t v2[vt];
#pragma unroll
  for (int j = 0; j < n1; j++) {
    const vec_in1 in = *reinterpret_cast<const vec_in1*>(
        data1 + (x + j * vt1) * static_cast<int64_t>(sizeof(in1_t)));
#pragma unroll
    for (int i = 0; i < vt1; i++) {
      v1[j * vt1 + i] = c10::convert<f_arg0_t>(in.val[i]);
    }
  }
#pragma unroll
  for (int j = 0; j < n2; j++) {
    const vec_in2 in = *reinterpret_cast<const vec_in2*>(
        data2 + (x + j * vt2) * static_cast<int64_t>(sizeof(in2_t)));
#pragma unroll
    for (int i = 0; i < vt2; i++) {
      v2[j * vt2 + i] = c10::convert<f_arg1_t>(in.val[i]);
    }
  }
  res_t vals[vt];
#pragma unroll
  for (int i = 0; i < vt; i++) {
    vals[i] = f(v1[i], v2[i]);
  }
#pragma unroll
  for (int j = 0; j < n0; j++) {
    vec_res out;
#pragma unroll
    for (int i = 0; i < vt0; i++) {
      out.val[i] = vals[j * vt0 + i];
    }
    *reinterpret_cast<vec_res*>(
        data0 + (x + j * vt0) * static_cast<int64_t>(sizeof(res_t))) = out;
  }
}

// Float output and the uncast float operand have the same vector width. Scan
// that width and the cast operand width independently, ending at natural
// alignment so offset tensors retain a specialized, correct fallback.
template <int vt_float, int vt_cast, typename cast_t, typename in1_t, typename in2_t, typename func_t, typename array_t>
static inline bool try_launch_1_2_cast_vt_scan(const func_t& f, array_t data, int64_t numel) {
  constexpr int vt = vt_float > vt_cast ? vt_float : vt_cast;
  if (numel % vt == 0 &&
      is_vector_access_aligned<float, vt_float>(data[0]) &&
      is_vector_access_aligned<in1_t, vt_float>(data[1]) &&
      is_vector_access_aligned<cast_t, vt_cast>(data[2])) {
    constexpr int nt = 256;
    dim3 block(nt);
    dim3 grid((numel + static_cast<int64_t>(nt) * vt - 1) /
              (static_cast<int64_t>(nt) * vt));
    auto stream = at::cuda::getCurrentCUDAStream();
    unrolled_elementwise_kernel_1_2_contiguous_castin2<
        vt_float, vt_float, vt_cast, nt, float, in1_t, in2_t, func_t>
        <<<grid, block, 0, stream>>>(numel, data[0], data[1], data[2], f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
  if constexpr (vt_cast > 1) {
    return try_launch_1_2_cast_vt_scan<vt_float, vt_cast / 2, cast_t, in1_t, in2_t, func_t>(
        f, data, numel);
  } else if constexpr (vt_float > 1) {
    return try_launch_1_2_cast_vt_scan<
        vt_float / 2, per_op_vector_width<cast_t>(), cast_t, in1_t, in2_t, func_t>(f, data, numel);
  } else {
    return false;
  }
}

template <typename in2_t, typename func_t, typename array_t>
static inline bool launch_unrolled_elementwise_kernel_1_2_contiguous_castin2(
    TensorIteratorBase& iter, const func_t& f, array_t data, int64_t numel) {
  return try_launch_1_2_cast_vt_scan<4, per_op_vector_width<in2_t>(), in2_t, float, in2_t, func_t>(
      f, data, numel);
}

// Mirrors the castin2 scan with the cast operand in slot one:
// the float operand width and the cast operand width are scanned independently,
// ending at natural alignment so offset tensors keep a specialized, correct
// fallback instead of dropping to the legacy unrolled path.
template <int vt_float, int vt_cast, typename cast_t, typename func_t, typename array_t>
static inline bool try_launch_1_2_castin1_vt_scan(const func_t& f, array_t data, int64_t numel) {
  constexpr int vt = vt_float > vt_cast ? vt_float : vt_cast;
  if (numel % vt == 0 &&
      is_vector_access_aligned<float, vt_float>(data[0]) &&
      is_vector_access_aligned<cast_t, vt_cast>(data[1]) &&
      is_vector_access_aligned<float, vt_float>(data[2])) {
    constexpr int nt = 256;
    dim3 block(nt);
    dim3 grid((numel + static_cast<int64_t>(nt) * vt - 1) /
              (static_cast<int64_t>(nt) * vt));
    auto stream = at::cuda::getCurrentCUDAStream();
    unrolled_elementwise_kernel_1_2_contiguous_castin2<
        vt_float, vt_cast, vt_float, nt, float, cast_t, float, func_t>
        <<<grid, block, 0, stream>>>(numel, data[0], data[1], data[2], f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
  if constexpr (vt_cast > 1) {
    return try_launch_1_2_castin1_vt_scan<vt_float, vt_cast / 2, cast_t, func_t>(f, data, numel);
  } else if constexpr (vt_float > 1) {
    return try_launch_1_2_castin1_vt_scan<
        vt_float / 2, per_op_vector_width<cast_t>(), cast_t, func_t>(f, data, numel);
  } else {
    return false;
  }
}

template <typename in1_t, typename func_t, typename array_t>
static inline bool launch_unrolled_elementwise_kernel_1_2_contiguous_castin1(
    TensorIteratorBase& iter, const func_t& f, array_t data, int64_t numel) {
  return try_launch_1_2_castin1_vt_scan<4, per_op_vector_width<in1_t>(), in1_t, func_t>(
      f, data, numel);
}
