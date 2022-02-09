
#pragma once

#include <ATen/jit_macros.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/thread_constants.h>

#include <thrust/tuple.h>

#include <ATen/native/cuda/MemoryAccess.cuh>


namespace at { namespace native {

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIteratorBase& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template<typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  int idx = blockIdx.x;

  return_t results[thread_work_size()];
  args_t args[thread_work_size()];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}

}}  // namespace at::native

// Note:
// CUDA and ROCm get diverged in this PR:
//   https://github.com/pytorch/pytorch/pull/32383
// Because for some reason trying to enable vectorized
// memory access introduce regression on ROCm.

#if !defined(USE_ROCM)
  #include <ATen/native/cuda/CUDALoops.cuh>
#else
  #include <ATen/native/cuda/ROCmLoops.cuh>
#endif

namespace at { namespace native {

#if AT_USE_JITERATOR()
/* Note [Jiterator]
The "jiterator" simply just-in-time compiles the same kernels that
Loops.cuh (and CUDALoops.cuh) usually build. This reduces build time,
build size, and initial CUDA context size.

By default on non-Windows systems, it also caches compiled kernels in ~/.cache/torch/kernels.
This behavior is controlled with two environment variables:
  - USE_PYTORCH_KERNEL_CACHE, if set to zero then this will disable all cache use
  - PYTORCH_KERNEL_CACHE_PATH, if set specifies the folder to use for cached kernels

The jiterator currently has some limitations, however. It cannot:
  - handle math on complex datatypes
  - handle kernels with scalar parameters

These improvements will likely come soon.

For examples of how to use the jiterator see the i1 and gcd kernel
implementations, which pass jittable strings implementing their
operations instead of the typical CUDA functors.

To pass a runtime argument (similar to lambda captures in non-JIT kernels),
we need to pass to additional arguments to `jitted_gpu_kernel` by value.
Currently only primitive C++ types used for computation are valid.
The order of these extra arguments should be same as the order they appear
in kernel's function signature. (look at polygamma for example)

NOTE: One big restriction being that these arguments should be after the
arguments provided by TensorIterator. Eg. While capturing `n`, where
`scalar_t x` and `scalar_t y` are provided by TensorIterator,
* foo(scalar_t x, scalar_t y, int n) works!
* foo(int n, scalar_t x, scalar_y) doesn't work
* foo(scalar_t x, int n, scalar_y) doesn't work

*/

// Entrypoint for jitted GPU kernels.
// Only handles elementwise unary and binary kernels with a
//   common dtype and a single output.
// NOTE: this assumes the op's iterator has a common_dtype.
// NOTE: We use std::tuple instead of parameter pack
//  for `extra_args` due to following
// bug on older versions of clang
// https://bugs.llvm.org/show_bug.cgi?id=23029
template <
    char const* name,
    typename return_type,
    typename f_inputs_type,
    int arity,
    typename... Args>
void jitted_gpu_kernel(
    TensorIteratorBase& iter,
    const std::string& f,
    at::cuda::jit::BinaryFuncVariant scalar_pos =
        at::cuda::jit::BinaryFuncVariant::NoScalar,
    at::opmath_type<f_inputs_type> scalar_val = 0,
    std::tuple<Args...> extra_args = std::make_tuple()) {
  // TODO: much of preamble is common to both jitted_gpu_kernel and gpu_kernel
  //   Maybe it could be refactored?
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel<name, return_type, f_inputs_type, arity>(
          sub_iter, f, scalar_pos, scalar_val, extra_args);
    }

    return;
  }

  // Computes if dynamic casting is needed
  // Dynamic casting is needed if an input's dtype differs from the common dtype
  //   or if the result dtype differs from the output's dtype
  // Note: this is intentionally divergent from calling needs_dynamic_casting,
  //   which is more general and inspects a lambda to determine if dynamic
  //   casting is needed.
  bool needs_dynamic_casting = false;

  // Checks output
  const ScalarType return_scalar_type = c10::CppTypeToScalarType<return_type>::value;
  const auto dtype0 = iter.dtype(0);
  if (dtype0 != return_scalar_type) {
    needs_dynamic_casting = true;
  }

  // Checks input(s)
  const ScalarType inputs_scalar_type = c10::CppTypeToScalarType<f_inputs_type>::value;
  for (auto i = decltype(arity){1}; i < (arity + 1); ++i) {
    const auto dtypei = iter.dtype(i);
    if (dtypei != inputs_scalar_type) {
      needs_dynamic_casting = true;
      break;
    }
  }
  if (scalar_pos == at::cuda::jit::BinaryFuncVariant::NoScalar) {
    // NOTE: With `scalar_pos=NoScalar`,`scalar_val` is not used
    // for computation in the generated code and hence we pass a dummy
    // value of `0`.
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::NoScalar>(
        iter, f, needs_dynamic_casting, /*scalar_val=*/0, extra_args);
  } else if (scalar_pos == at::cuda::jit::BinaryFuncVariant::RhsScalar) {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::RhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);

  } else {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::LhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);
  }
}

// TODO: support runtime state capture similar to `jitted_gpu_kernel`.
template <char const *name, typename return_type, typename f_inputs_type>
void opmath_jitted_gpu_kernel_with_scalars(TensorIteratorBase& iter, const std::string& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);
  //currently jiterator only handles binary functions where both inputs are of the same type (f_inputs_type)
  using opmath_t = at::opmath_type<f_inputs_type>;
  if (iter.is_cpu_scalar(1)) {
    auto scalar_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::LhsScalar, scalar_val);
  } else if (iter.is_cpu_scalar(2)) {
    auto scalar_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::RhsScalar, scalar_val);
  } else {
    jitted_gpu_kernel<name, return_type, f_inputs_type, 2>(iter, f);
  }
}
#endif // AT_USE_JITERATOR()

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl(iter, f);
}

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  __device__ return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  AUnaryFunctor(func_t f_, opmath_arg1_t a_): f(f_), a(a_) {}
  private:
    func_t f;
    opmath_arg1_t a;
};

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  __device__ return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  BUnaryFunctor(func_t f_, opmath_arg2_t b_): f(f_), b(b_) {}
  private:
    func_t f;
    opmath_arg2_t b;
};

// Though seemingly noop, this inserts casts from arg1_t to func_t's type
// (which may be higher precision), as well as casts to return_t
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  __device__ return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_): f(f_) {}
  private:
    func_t f;
};

// Unlike gpu_kernel_with_scalars, this allows you to pass a func_t which
// accepts inputs at higher precision (typically opmath_t), but then
// ensure that we load from memory at the correct precision (scalar_t)
// to avoid expensive loads.  For the whole sordid story see
// https://dev-discuss.pytorch.org/t/cuda-loops-case-study-code-generation-vs-templates/302
template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t, typename func_t>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    gpu_kernel(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel(iter, bf);
  } else {
    gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

// Legacy variant that assumes that func_t has the correct types
// that we expect to load from memory
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

namespace { // functions for `gpu_kernel_multiple_outputs`.

// check the return type is `thrust::tuple`, not `std::tuple`.
template <typename T> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<thrust::tuple<T...>>: std::true_type {};

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int remaining = N - block_work_size() * blockIdx.x;
  elementwise_kernel_helper(f, memory::policies::multi_outputs_unroll<array_t, inp_calc_t, out_calc_t, num_outputs>(data, remaining, ic, oc));
}

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  static_assert(is_tuple<output_t>::value, "f's return type must be `thrust::tuple`");
  constexpr int num_outputs = thrust::tuple_size<output_t>::value;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  if (iter.is_contiguous()) {
    auto input_calc = TrivialOffsetCalculator<num_inputs>();
    auto output_calc = TrivialOffsetCalculator<num_outputs>();
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  } else {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  }
}
} // namespace

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  gpu_kernel_multiple_outputs_impl(iter, f);
}

}} //namespace at::native
