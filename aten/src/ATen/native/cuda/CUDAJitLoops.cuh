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

#include <type_traits>
#include <tuple>
#include <mutex>

namespace at {
namespace native {

namespace {

template <typename Tuple, std::size_t... I>
constexpr auto tuple_to_array_helper(Tuple& t, std::index_sequence<I...> seq) {
    constexpr auto size = seq.size();
    (void)t; // warning : unused parameter when tuple is empty.
    return std::array<void*, size>{static_cast<void*>(&std::get<I>(t))...};
}

// Helper function convert tuple to std::array<void*, N>
// for passing the arguments to CUDA Kernel
// NOTE: We capture tuple by reference,
// so the pointers in returned array are only valid
// till tuple is alive.
template <typename ...Args>
constexpr auto tuple_to_array(std::tuple<Args...>& extra_args) {
    constexpr auto tuple_size = sizeof...(Args);
    return tuple_to_array_helper(extra_args, std::make_index_sequence<tuple_size>{});
}

// Helper function to return a vector<string>
// corresponding to the type of the arguments in parameter pack.
template <typename... Args>
c10::SmallVector<std::string> get_extra_args_typenames() {
  return {at::cuda::jit::typeName<Args>()...};
}

} // namespace

template<char const *name,
         typename result_type,
         typename f_inputs_type,
         at::cuda::jit::BinaryFuncVariant scalar_pos,
         typename array_t,
         typename inp_calc_t,
         typename out_calc_t,
         typename loader_t,
         typename storer_t,
         typename ... Args>
static inline void launch_jitted_unrolled_kernel(
  DeviceIndex dev_idx, int64_t N, const std::string& f, array_t data,
  inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s, bool contiguous,
  at::opmath_type<f_inputs_type> scalar_val,
  std::tuple<Args...> extra_args) {

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  const int64_t grid = (N + block_work_size() - 1) / block_work_size();

  static std::mutex _jiterator_mutex;
  static std::vector<at::cuda::jit::NvrtcFunction> fns(c10::cuda::device_count());

  at::cuda::jit::NvrtcFunction* fn_ptr = &fns[dev_idx];
  if (!fn_ptr->function) {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    if (!fn_ptr->function) {
      constexpr int nTensors = array_t::size();
      constexpr bool dynamic_casting = !std::is_same<decltype(l),
                                                     memory::LoadWithoutCast>() || !std::is_same<decltype(s),
                                                     memory::StoreWithoutCast>();
      std::string string_name{name};
      std::string f_inputs_type_str = at::cuda::jit::typeName<f_inputs_type>();
      std::string compute_type_str = at::cuda::jit::typeName<at::opmath_type<f_inputs_type>>();
      std::string result_type_str = at::cuda::jit::typeName<result_type>();
      c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames<Args...>();
      auto code = at::cuda::jit::generate_code(nTensors, f, string_name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               contiguous, dynamic_casting, scalar_pos, extra_args_types);
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, name);
    }
  }

  // pack args for kernel launch
  constexpr int kernel_args = 7;
  // size of `extra_args` is known at compile-time
  constexpr auto extra_args_size = sizeof...(Args);
  void* args[kernel_args + extra_args_size];
  args[0] = static_cast<void*>(&N);
  args[1] = static_cast<void*>(&data);
  args[2] = static_cast<void*>(&ic);
  args[3] = static_cast<void*>(&oc);
  args[4] = static_cast<void*>(&l);
  args[5] = static_cast<void*>(&s);
  args[6] = static_cast<void*>(&scalar_val);

  auto extra_args_array = tuple_to_array(extra_args);
  for (const auto i : c10::irange(extra_args_size)) {
    // since 7 slots are already filled in `args`
    args[i + 7] = extra_args_array[i];
  }

  at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<
  char const *name,
  typename result_type,
  typename f_inputs_type,
  int arity,
  at::cuda::jit::BinaryFuncVariant scalar_pos,
  typename array_t, typename ... Args>
static inline void launch_jitted_vectorized_kernel(DeviceIndex dev_idx, int64_t N, const std::string& f, array_t data,
at::opmath_type<f_inputs_type> scalar_val, std::tuple<Args...> extra_args) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  const int64_t grid = (N + block_work_size() - 1) / block_work_size();
  const int vec_size = memory::jitted_can_vectorize_up_to<result_type, f_inputs_type, arity>(data);

  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  //   fn_ptr is set to the appropriate function based on the vec size and GPU used
  // TODO: Memory use can probably be optimized by re-using kernels across GPUs with
  //   the same compute capability
  static std::mutex _jiterator_mutex;
  static std::vector<at::cuda::jit::NvrtcFunction> fns4(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns2(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns1(c10::cuda::device_count());


  at::cuda::jit::NvrtcFunction* fn_ptr;
  if (vec_size == 4) {
    fn_ptr = &fns4[dev_idx];
  } else if (vec_size == 2) {
    fn_ptr = &fns2[dev_idx];
  } else if (vec_size ==1) {
    fn_ptr = &fns1[dev_idx];
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected vec_size for jitter vectorized kernel");
  }

  bool vectorized = vec_size > 1;

  if (!fn_ptr->function) {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    if (!fn_ptr->function) { // cache miss!

      // Generates program
      constexpr int nTensors = array_t::size();
      std::string string_name{name};
      std::string f_inputs_type_str = at::cuda::jit::typeName<f_inputs_type>();
      std::string compute_type_str = at::cuda::jit::typeName<at::opmath_type<f_inputs_type>>();
      std::string result_type_str = at::cuda::jit::typeName<result_type>();
      c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames<Args...>();
      auto code = at::cuda::jit::generate_code(nTensors, f, string_name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               /*contiguous=*/true, /*dynamic_casting=*/false,
                                               scalar_pos,
                                               extra_args_types,
                                               vectorized, vec_size);
      std::string kernel_name = vectorized ? string_name + "_vectorized" + std::to_string(vec_size) : string_name;

      // Acquires the program
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
    }
  }

  // size of `extra_args` is known at compile-time
  constexpr auto extra_args_size = sizeof...(Args);
  auto extra_args_array = tuple_to_array(extra_args);

  if (vectorized) {
    // pack args for kernel launch
    constexpr int kernel_args = 3;
    void* args[kernel_args + extra_args_size];
    args[0] = static_cast<void*>(&N);
    args[1] = static_cast<void*>(&data);
    args[2] = static_cast<void*>(&scalar_val);

    for (const auto i : c10::irange(extra_args_size)) {
      // since 3 slots are already filled in `args`
      args[i + 3] = extra_args_array[i];
    }

    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto ic = TrivialOffsetCalculator<arity>();
    auto oc = TrivialOffsetCalculator<1>();
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();

    // pack args for kernel launch
    constexpr int kernel_args = 7;
    void* args[kernel_args + extra_args_size];
    args[0] = static_cast<void*>(&N);
    args[1] = static_cast<void*>(&data);
    args[2] = static_cast<void*>(&ic);
    args[3] = static_cast<void*>(&oc);
    args[4] = static_cast<void*>(&l);
    args[5] = static_cast<void*>(&s);
    args[6] = static_cast<void*>(&scalar_val);

    for (const auto i : c10::irange(extra_args_size)) {
      // since 7 slots are already filled in `args`
      args[i + 7] = extra_args_array[i];
    }
    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <char const *name, typename result_type, typename compute_type, int arity,
          at::cuda::jit::BinaryFuncVariant scalar_pos=at::cuda::jit::BinaryFuncVariant::NoScalar, typename ... Args>
void jitted_gpu_kernel_impl(TensorIteratorBase& iter, const std::string& f, const bool dynamic_casting, compute_type scalar_val, std::tuple<Args...> extra_args) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  constexpr int ntensors = arity + 1;
  at::detail::Array<char*, ntensors> data;
  for (auto i = decltype(ntensors){0}; i < ntensors; ++i) {
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
      launch_jitted_vectorized_kernel<name, result_type, compute_type, arity, scalar_pos>(
        iter.device().index(), numel, f, data, scalar_val, extra_args);
      return;
    }

    // Case 2: no dynamic casting and noncontiguous
    auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
      iter.device().index(), numel, f, data, input_offset_calculator,
      output_offset_calculator, loader, storer, contiguous, scalar_val, extra_args);
    return;
  }

  // Cases 3 and 4 are handled below
  // Both require construction of a storer (this asserts 1 output) and one or more loaders

  // Creates store cast to output (the zeroth tensor in TensorIterator)
  auto storer = memory::StoreWithCast(iter.dtype(0));

  // Creates load casts from inputs (note offset indexing into the iterators 1...n tensors)
  at::detail::Array<ScalarType, arity> dtypes;
  for (auto i = decltype(arity){0}; i < arity; ++i) {
    dtypes[i] = iter.dtype(i + 1);
  }
  auto loader = memory::LoadWithCast<arity>(dtypes);

  if (contiguous) {
    // Case 3: dynamic casting and contiguous
    auto input_offset_calculator = TrivialOffsetCalculator<arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
      iter.device().index(), numel, f, data, input_offset_calculator,
      output_offset_calculator, loader, storer, contiguous, scalar_val, extra_args);
    return;
  }

  // Case 4: dynamic casting and noncontiguous
  auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
  auto output_offset_calculator = make_output_offset_calculator(iter);
  launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
    iter.device().index(), numel, f, data, input_offset_calculator,
    output_offset_calculator, loader, storer, contiguous, scalar_val, extra_args);
}

}}  // at::native

#endif // AT_USE_JITERATOR()
