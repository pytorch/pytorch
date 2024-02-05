#pragma once

#include <ATen/ceil_div.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/detail/ElementwiseInvoke.h>
#include <ATen/detail/OffsetCalculator.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/thread_constants.h>

#include <ATen/native/xpu/sycl/SYCLHelpers.h> // SYCL context
#include <ATen/native/xpu/sycl/MemoryAccess.h>

// XXX:
#include <sycl/sycl.hpp>
#define sycl_queue() \
    sycl::queue(sycl::gpu_selector_v, {sycl::property::queue::in_order()})

namespace at { namespace native { namespace xpu {

template <int item_work_size, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[item_work_size];
  args_t args[item_work_size];

  // load
  policy.load(args);

  // compute
#pragma unroll
  for (int i = 0; i < item_work_size; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results);
}

template <int vec_size, typename func_t>
struct ElementwiseKernel {
  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int gid = item.get_global_linear_id();
  #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      if (gid < numel_) {
        f_(gid);
        gid += grpsz;
      }
    }
  };

  ElementwiseKernel(int numel, func_t f)
      : numel_(numel), f_(f) {}

 private:
  int numel_;
  func_t f_;
};

template <
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
struct UnrolledElementwiseKernel {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        loader_t,
        storer_t>(
        data_, remaining, ic_, oc_, l_, s_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseKernel(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc), l_(l), s_(s) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
  loader_t l_;
  storer_t s_;
};

template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
struct VectorizedElementwiseKernel {
  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int group_work_size = vec_size * grpsz;
    int remaining = numel_ - grpid * group_work_size;

    // ic_
    if (remaining < group_work_size) {
      auto oc = TrivialOffsetCalculator<1>();
      auto l = at::native::memory::LoadWithoutCast();
      auto s = at::native::memory::StoreWithoutCast();
      auto policy = at::native::memory::policies::unroll<
          vec_size,
          array_t,
          decltype(ic_),
          decltype(oc),
          at::native::memory::LoadWithoutCast,
          at::native::memory::StoreWithoutCast>(
              data_,
              remaining,
              ic_,
              oc,
              l,
              s,
              lid,
              grpid,
              grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    } else {
      auto policy = at::native::memory::policies::
          vectorized<vec_size, array_t, in_calc_t>(
              data_, ic_, lid, grpid, grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    }
  }

  VectorizedElementwiseKernel(
      int numel,
      const func_t f,
      array_t data,
      in_calc_t ic)
      : numel_(numel), f_(f), data_(data), ic_(ic) {}

 private:
  int numel_;
  const func_t f_;
  array_t data_;
  in_calc_t ic_;
};

template <
    int num_outputs,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t>
struct UnrolledElementwiseForMultiOutputsKernel {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item_id) const {
    int grpsz = item_id.get_local_range(0);
    int grpid = item_id.get_group(0);
    int lid = item_id.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::multi_outputs_unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        num_outputs>(data_, remaining, ic_, oc_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseForMultiOutputsKernel(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
};

template <
    int num_outputs,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t>
struct UnrolledElementwiseKernelForMultiOutputs {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::multi_outputs_unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        num_outputs>(data_, remaining, ic_, oc_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseKernelForMultiOutputs(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
};

template <
    typename arg0_t,
    int ntensors,
    typename offset_calc_t,
    typename func_t>
struct LegacyKernelScalarFunctor {
  void operator()(int idx) const {
    auto offsets = offset_calc_.get(idx);
    arg0_t* out = (arg0_t*)(data_[0] + offsets[0]);
    *out = invoke(f_, &data_.data[1], &offsets.data[1], 1);
  }

  LegacyKernelScalarFunctor(
      at::detail::Array<char*, ntensors> data,
      offset_calc_t offset_calc,
      const func_t f)
      : data_(data), offset_calc_(offset_calc), f_(f) {}

 private:
  at::detail::Array<char*, ntensors> data_;
  offset_calc_t offset_calc_;
  const func_t f_;
};

template <
    typename arg0_t,
    int ntensors,
    typename offset_calc_t,
    typename func_t>
struct LegacyKernelWithCastScalarFunctor {
  void operator()(int idx) const {
    auto offsets = offset_calc_.get(idx);
    void* out = data_[0] + offsets[0];
    arg0_t result =
        invoke(f_, &data_.data[1], &offsets.data[1], &dtypes_.data[1], 1);
    c10::cast_and_store<arg0_t>(dtypes_[0], out, result);
  }

  LegacyKernelWithCastScalarFunctor(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<ScalarType, ntensors> dtypes,
      offset_calc_t offset_calc,
      const func_t f)
      : data_(data), dtypes_(dtypes), offset_calc_(offset_calc), f_(f) {}

 private:
  at::detail::Array<char*, ntensors> data_;
  at::detail::Array<ScalarType, ntensors> dtypes_;
  offset_calc_t offset_calc_;
  const func_t f_;
};

template <int vec_size, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto ker = ElementwiseKernel<vec_size, func_t>(N, f);

  int wg_sz = 1024; // dpcppMaxWorkItemsPerEU();
  int num_wg = ceil_div<int>(N, wg_sz * vec_size);
  sycl::kernel_submit(wg_sz * num_wg, wg_sz, sycl_queue(), ker);
}

template <
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto ker = UnrolledElementwiseKernel(N, f, data, ic, oc, l, s);
  using ker_t = decltype(ker);

  auto wg_sz = 1024; // dpcppMaxWorkItemsPerEU();
  int num_wg = ceil_div<int>(N, wg_sz * ker_t::item_work_size);
  sycl::kernel_submit(wg_sz * num_wg, wg_sz, sycl_queue(), ker);
}

template <typename func_t, typename array_t, typename in_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t input_calc,
    int vec_size) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  // XXX
  auto wg_sz = 1024; // dpcppMaxWorkItemsPerEU();

#define VEC_KER(vec_size)                                                    \
  {                                                                          \
    auto ker =                                                               \
        VectorizedElementwiseKernel<vec_size, func_t, array_t, in_calc_t>(   \
            N, f, data, input_calc);                                         \
    int num_wg = ceil_div<int>(N, wg_sz * vec_size);                         \
    sycl::kernel_submit(wg_sz * num_wg, wg_sz, sycl_queue(), ker);           \
  }

  switch (vec_size) {
    case 8:
      VEC_KER(8);
      break;
    case 4:
      VEC_KER(4);
      break;
    case 2:
      VEC_KER(2);
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();

      auto ker = UnrolledElementwiseKernel(
          N, f, data, input_calc, output_calc, loader, storer);
      using ker_t = decltype(ker);

      int num_wg = ceil_div<int>(N, wg_sz * ker_t::item_work_size);
      sycl::kernel_submit(wg_sz * num_wg, wg_sz, sycl_queue(), ker);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template <int num_outputs, typename func_t, typename array_t, typename in_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, in_calc_t ic, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto ker = UnrolledElementwiseForMultiOutputsKernel<
      num_outputs,
      func_t,
      array_t,
      in_calc_t,
      out_calc_t>(N, f, data, ic, oc);
  using ker_t = decltype(ker);

  int wg_sz = 1024; // dpcppMaxWorkItemsPerEU();
  int num_wg = ceil_div<int>(N, ker_t::item_work_size * wg_sz);
  sycl::kernel_submit(wg_sz * num_wg, wg_sz, sycl_queue(), ker);
}

template <typename func_t, typename data_t>
static inline bool can_vectorize_for_non_contigouous(
    TensorIteratorBase& iter,
    const data_t& data,
    int& vec_size) {
  // Only optimizing non conitguous cases here
  if (iter.is_contiguous())
    return false;
  // Fastest moving dimension should be contiguous
  if (!iter.has_contiguous_first_dim())
    return false;
  // Output should be contiguous
  if (!iter.tensor(0).is_contiguous())
    return false;
  vec_size = memory::can_vectorize_up_to<func_t>(data);
  if (vec_size <= 1)
    return false;
  for (int i = 0; i < iter.ntensors(); i++) {
    // Checking each row has enough alignments
    auto strides = iter.strides(i);
    for (int dim = 1; dim < strides.size(); dim++) {
      int64_t base_of_row = strides[dim];
      int64_t vec_size_in_bytes = strides[0] * vec_size;
      while (base_of_row % vec_size_in_bytes) {
        vec_size >>= 1;
        if (vec_size <= 1)
          return false;
      }
    }
  }
  return vec_size > 1;
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  int vec_size;
  if (contiguous) {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    vec_size = memory::can_vectorize_up_to<func_t>(data);
    launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
    return;
  } else if (can_vectorize_for_non_contigouous<func_t>(iter, data, vec_size)) {
    auto input_calc = make_input_offset_calculator<traits::arity>(iter);
    launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
    return;
  }

  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) > 4 ? 2 : 4;
  launch_legacy_kernel<unroll_factor>(
    numel, LegacyKernelScalarFunctor<
        arg0_t, ntensors, decltype(offset_calc), func_t>(data, offset_calc, f));
}

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f);
  }
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
  } else {
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    constexpr int unroll_factor = sizeof(arg0_t) > 4 ? 2 : 4;
    launch_legacy_kernel<unroll_factor>(
        numel,
        LegacyKernelWithCastScalarFunctor<
            arg0_t, ntensors, decltype(offset_calc), func_t>(
                data, dtypes, offset_calc, f));
  }
}

template <typename func_t>
void gpu_kernel_nocast(TensorIteratorBase& iter, const func_t& f) {

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
      gpu_kernel_nocast(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_nocast(iter, f);
}

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    // XXX is_cuda()
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
  return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  AUnaryFunctor(func_t f_, opmath_arg1_t a_): f(f_), a(a_) {}
  private:
    func_t f;
    opmath_arg1_t a;
};

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  BUnaryFunctor(func_t f_, opmath_arg2_t b_): f(f_), b(b_) {}
  private:
    func_t f;
    opmath_arg2_t b;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_): f(f_) {}
  private:
    func_t f;
};

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
    // XXX:
    // const OptionalDeviceGuard device_guard(iter.device(1));
    gpu_kernel(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel(iter, bf);
  } else {
    gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same<opmath_arg_t, typename traits::template arg<1>::type>::value,
                "f is not symmetric");

  // XXX:
  // OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);

    // XXX:
    // device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f);
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

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  constexpr int num_outputs = std::tuple_size<output_t>::value;
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

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
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

}}} //namespace at::native::xpu
