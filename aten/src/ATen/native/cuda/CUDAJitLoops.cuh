#pragma once
#include <ATen/jit_macros.h>

// Jiterator functions are guarded behind this macro
#if AT_USE_JITERATOR()

#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/thread_constants.h>

#include <ATen/native/cuda/Loops.cuh>

#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/util/SmallBuffer.h>

#include <array>
#include <initializer_list>
#include <type_traits>
#include <tuple>
#include <mutex>

namespace at::native {

template <typename Tuple, std::size_t... I>
// warning : unused parameter when tuple is empty.
constexpr auto tuple_to_array_helper(const Tuple& t [[maybe_unused]], std::index_sequence<I...> seq) {
    constexpr auto size = seq.size();
    return std::array<const void*, size>{static_cast<const void*>(&std::get<I>(t))...};
}

// Helper function convert tuple to std::array<const void*, N>
// for passing the arguments to CUDA Kernel
// NOTE: We capture tuple by reference,
// so the pointers in returned array are only valid
// till tuple is alive.
template <typename ...Args>
constexpr auto tuple_to_array(const std::tuple<Args...>& extra_args) {
    constexpr auto tuple_size = sizeof...(Args);
    return tuple_to_array_helper(extra_args, std::make_index_sequence<tuple_size>{});
}

struct JittedVecKernelCache {
  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  at::cuda::jit::NvrtcFunction vec1;
  at::cuda::jit::NvrtcFunction vec2;
  at::cuda::jit::NvrtcFunction vec4;
#ifdef USE_ROCM
  at::cuda::jit::NvrtcFunction vec8;
  at::cuda::jit::NvrtcFunction vec16;
#endif

};

struct JittedKernelVariantCache {
  JittedVecKernelCache vec;
  at::cuda::jit::NvrtcFunction noncontiguous;
  at::cuda::jit::NvrtcFunction dynamic_contiguous;
  at::cuda::jit::NvrtcFunction dynamic_noncontiguous;
};

inline c10::SmallBuffer<const void*, 64> pack_kernel_args(
    std::initializer_list<const void*> args,
    c10::ArrayRef<const void*> extra_args) {
  c10::SmallBuffer<const void*, 64> ret(args.size() + extra_args.size());
  std::copy(args.begin(), args.end(), ret.data());
  std::copy(extra_args.begin(), extra_args.end(), ret.data() + args.size());
  return ret;
}

template<typename array_t,
         typename inp_calc_t,
         typename out_calc_t,
         typename loader_t,
         typename storer_t>
void launch_jitted_unrolled_kernel(
    std::mutex &jiterator_mutex,
    at::cuda::jit::NvrtcFunction &fn_cache,
    const at::cuda::jit::KernelDescriptor &desc,
    int64_t N,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s,
    bool contiguous,
    at::cuda::jit::BinaryFuncVariant scalar_pos,
    const void* scalar_val,
    c10::ArrayRef<const void*> extra_args) {

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int tws = at::cuda::jit::calc_thread_work_size(desc.nInputs, desc.nOutputs, desc.f_inputs_type, desc.result_type);
  int bws = tws * num_threads();
  //casting result to int is always safe, intermediate is int64 and won't overflow
  const uint32_t grid = (N + bws - 1) / bws;

  if (!fn_cache.function) {
    const std::lock_guard<std::mutex> lock{jiterator_mutex};
    if (!fn_cache.function) {
      constexpr bool dynamic_casting = !std::is_same<decltype(l), memory::LoadWithoutCast>() ||
                                       !std::is_same<decltype(s), memory::StoreWithoutCast>();
      auto code = at::cuda::jit::generate_code(
          desc, contiguous, dynamic_casting, scalar_pos, tws);
      fn_cache = at::cuda::jit::jit_pwise_function(code, desc.name);
    }
  }

  auto args = pack_kernel_args({&N, &data, &ic, &oc, &l, &s, scalar_val}, extra_args);
  at::cuda::jit::launch_jitted_pwise_function(fn_cache, args.data(), {grid, 1u, 1u},
  {num_threads(), 1u, 1u});
}

template<int arity, typename array_t>
void launch_jitted_vectorized_kernel(
    std::mutex &jiterator_mutex, JittedVecKernelCache &fn_cache,
    const at::cuda::jit::KernelDescriptor &desc, int64_t N, array_t data,
    at::cuda::jit::BinaryFuncVariant scalar_pos,
    const void *scalar_val, c10::ArrayRef<const void*> extra_args) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int tws = at::cuda::jit::calc_thread_work_size(desc.nInputs, desc.nOutputs, desc.f_inputs_type, desc.result_type);
  int bws = tws * num_threads();
  // N is still int64_t for the computation, but it's always safe to cast result to int
  const uint32_t grid = (N + bws - 1) / bws;

  int vec_size = at::cuda::jit::can_vectorize_up_to(
      desc, c10::ArrayRef<char*>(data.data(), data.size()));

  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  //   fn_ptr is set to the appropriate function based on the vec size and GPU used
  at::cuda::jit::NvrtcFunction* fn_ptr = nullptr;

#ifdef USE_ROCM
  if (vec_size == 16) {
    fn_ptr = &fn_cache.vec16;
  } else if (vec_size == 8) {
    fn_ptr = &fn_cache.vec8;
  } else
#endif
  if (vec_size == 4) {
    fn_ptr = &fn_cache.vec4;
  } else if (vec_size == 2) {
    fn_ptr = &fn_cache.vec2;
  } else if (vec_size ==1) {
    fn_ptr = &fn_cache.vec1;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected vec_size for jitter vectorized kernel");
  }

  bool vectorized = vec_size > 1;

  if (!fn_ptr->function) {
    const std::lock_guard<std::mutex> lock{jiterator_mutex};
    if (!fn_ptr->function) { // cache miss!

      // Generates program
      auto code = at::cuda::jit::generate_code(
          desc, /*contiguous=*/true, /*dynamic_casting=*/false,
          scalar_pos, tws, vectorized, vec_size);
      std::string kernel_name = vectorized ? desc.name + "_vectorized" + std::to_string(vec_size) : desc.name;

      // Acquires the program
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
    }
  }

  if (vectorized) {
    auto args = pack_kernel_args({&N, &data, scalar_val}, extra_args);
    at::cuda::jit::launch_jitted_pwise_function(
        *fn_ptr, args.data(), {grid, 1u, 1u}, {num_threads(), 1u, 1u});
  } else {
// NVCC complains about unused variables l and s.
// It should be false positive in most cases, so we suppress the warnings.
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177
    auto ic = TrivialOffsetCalculator<arity>();
    auto oc = TrivialOffsetCalculator<1>();
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();

    auto args = pack_kernel_args(
        {&N, &data, &ic, &oc, &l, &s, scalar_val}, extra_args);
    at::cuda::jit::launch_jitted_pwise_function(
        *fn_ptr, args.data(), {grid, 1u, 1u}, {num_threads(), 1u, 1u});
#pragma nv_diagnostic pop
  }
}

template <int arity>
void jitted_gpu_kernel_generic(
    std::mutex &jiterator_mutex,
    JittedKernelVariantCache &cache,
    const at::cuda::jit::KernelDescriptor &desc,
    at::cuda::jit::BinaryFuncVariant scalar_pos,
    c10::ArrayRef<const void*> extra_args,
    TensorIteratorBase& iter,
    const bool dynamic_casting,
    const void *scalar_val) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  constexpr int ntensors = arity + 1;
  std::array<char*, ntensors> data;
  for (auto i : c10::irange(ntensors)) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  // Decides which of 4 kernel types to launch
  // Variations are:
  //   - Case 1: no dynamic casting and contiguous
  //   - Case 2: no dynamic casting and noncontiguous
  //   - Case 3: dynamic casting and contiguous
  //   - Case 4: dynamic casting and noncontiguous
  // These cases align with the non-jitted CUDALoops.cuh cases in gpu_kernel_impl

  if (!dynamic_casting) {
    if (contiguous) {
      // Case 1: no dynamic casting and contiguous
      launch_jitted_vectorized_kernel<arity>(
          jiterator_mutex, cache.vec, desc,
          numel, data, scalar_pos, scalar_val, extra_args);
      return;
    }

    // Case 2: no dynamic casting and noncontiguous
    auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    launch_jitted_unrolled_kernel(
        jiterator_mutex, cache.noncontiguous, desc, numel, data,
        input_offset_calculator, output_offset_calculator, loader,
        storer, contiguous, scalar_pos, scalar_val, extra_args);
    return;
  }

  // Cases 3 and 4 are handled below
  // Both require construction of a storer (this asserts 1 output) and one or more loaders

  // Creates store cast to output (the zeroth tensor in TensorIterator)
  auto storer = memory::StoreWithCast<1>(iter);

  // Creates load casts from inputs (note offset indexing into the iterators 1...n tensors)
  auto loader = memory::LoadWithCast<arity>(iter);

  if (contiguous) {
    // Case 3: dynamic casting and contiguous
    auto input_offset_calculator = TrivialOffsetCalculator<arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_jitted_unrolled_kernel(
        jiterator_mutex, cache.dynamic_contiguous, desc, numel, data, input_offset_calculator,
        output_offset_calculator, loader, storer, contiguous, scalar_pos, scalar_val, extra_args);
    return;
  }

  // Case 4: dynamic casting and noncontiguous
  auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
  auto output_offset_calculator = make_output_offset_calculator(iter);
  launch_jitted_unrolled_kernel(
      jiterator_mutex, cache.dynamic_noncontiguous, desc, numel, data, input_offset_calculator,
      output_offset_calculator, loader, storer, contiguous, scalar_pos, scalar_val, extra_args);
}

// NOTE: static to reduce chances of name collision.
template <
    char const* name,
    typename result_type,
    typename f_inputs_type,
    int arity,
    at::cuda::jit::BinaryFuncVariant scalar_pos =
        at::cuda::jit::BinaryFuncVariant::NoScalar,
    typename... ExtraArgs>
static void jitted_gpu_kernel_impl(
    TensorIteratorBase& iter,
    const std::string &f,
    const bool dynamic_casting,
    at::opmath_type<f_inputs_type> scalar_val,
    const std::tuple<ExtraArgs...>& extra_args) {

  // TODO: Memory use can probably be optimized by re-using kernels across GPUs with
  //   the same compute capability
  static std::mutex jiterator_mutex;
  static std::vector<JittedKernelVariantCache> device_caches(c10::cuda::device_count());

  constexpr int nInputs = arity;
  constexpr int nOutputs = 1;  // TODO: Support more than 1 output
  static const auto desc = at::cuda::jit::make_kernel_descriptor<
    result_type, f_inputs_type, ExtraArgs...>(name, f, nInputs, nOutputs);

  auto &cache = device_caches[iter.device().index()];
  auto extra_args_array = tuple_to_array(extra_args);
  return jitted_gpu_kernel_generic<arity>(
      jiterator_mutex,
      cache,
      desc,
      scalar_pos,
      extra_args_array,
      iter,
      dynamic_casting,
      &scalar_val
    );
}

}  // at::native

#endif // AT_USE_JITERATOR()
