#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <type_traits>
#include <tuple>
#include <iostream>
#include <mutex>

#include <ATen/jit_macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <c10/util/C++17.h>


#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) \
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif


namespace at { namespace native {

template<int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;

  if (remaining < block_work_size()) {  // if this block handles the reminder, just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<array_t, decltype(input_calc), decltype(output_calc),
                                           memory::LoadWithoutCast, memory::StoreWithoutCast>(
      data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {  // if this block has a full `block_work_size` data to handle, use vectorized memory access
    elementwise_kernel_helper(f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
                                            inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  int remaining = N - block_work_size() * blockIdx.x;
  auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
  case 4:
    vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 1: {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  }
  default:
    TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

// Jiterator functions are guarded behind this macro
#if AT_USE_JITERATOR()

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
#endif // AT_USE_JITERATOR()

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
static inline void launch_unrolled_kernel(int64_t N, const func_t& f, array_t data,
                                          inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(*(typename traits::template arg<INDEX>::type*)(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i,
            std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(dtypes[I], data[I] + i * strides[I])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}


template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
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
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);

  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel(numel, f, data);
    } else {
        auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
        launch_legacy_kernel<128,unroll_factor>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke(f, &data.data[1], &offsets.data[1], 1);
        });
    }
  } else {
    if (contiguous) {
      at::detail::Array<ScalarType, traits::arity> dtypes;
      for (int i = 0; i < traits::arity; i++) {
        dtypes[i] = iter.dtype(i + 1);
      }
      auto loader = memory::LoadWithCast<traits::arity>(dtypes);
      auto storer = memory::StoreWithCast(iter.dtype(0));
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      at::detail::Array<ScalarType, ntensors> dtypes;
      for (int i = 0; i < ntensors; i++) {
        dtypes[i] = iter.dtype(i);
      }
      auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
      launch_legacy_kernel<128, 4>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      });
    }
  }
}

}} // namespace at::native
