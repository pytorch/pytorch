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

#include <array>
#include <tuple>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeCast.h>

#ifndef USE_ROCM
#include <cuda/cmath>
#endif

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type)                       \
  static_assert(                                              \
      __nv_is_extended_host_device_lambda_closure_type(type), \
      #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

namespace at::native {

template <typename func_t, int cc_major, int cc_minor, typename = void>
struct get_is_simple {
  static constexpr bool value = false;
};

template <typename func_t, int cc_major, int cc_minor>
struct get_is_simple<func_t, cc_major, cc_minor, std::void_t<
      decltype(func_t::template is_simple<cc_major, cc_minor>)>
    > {
  static constexpr bool value = func_t::template is_simple<cc_major, cc_minor>;
};

template <int cc_major_, int cc_minor_>
struct cc_helper {
  static constexpr int major = cc_major_;
  static constexpr int minor = cc_minor_;
};

// Runtime CC dispatch: defines `using cc = cc_helper<M,N>` then
// executes __VA_ARGS__ in that scope, for each supported CC.
#define AT_DISPATCH_CC(major, minor, ...)                                     \
  if ((major) == 12 && (minor) == 1) {                                        \
    using cc = cc_helper<12, 1>; __VA_ARGS__;                                 \
  } else if ((major) == 12 && (minor) == 0) {                                 \
    using cc = cc_helper<12, 0>; __VA_ARGS__;                                 \
  } else if ((major) == 11 && (minor) == 0) {                                 \
    using cc = cc_helper<11, 0>; __VA_ARGS__;                                 \
  } else if ((major) == 10 && (minor) == 3) {                                 \
    using cc = cc_helper<10, 3>; __VA_ARGS__;                                 \
  } else if ((major) == 10 && (minor) == 0) {                                 \
    using cc = cc_helper<10, 0>; __VA_ARGS__;                                 \
  } else if ((major) == 9 && (minor) == 0) {                                  \
    using cc = cc_helper<9, 0>; __VA_ARGS__;                                  \
  } else if ((major) == 8 && (minor) == 9) {                                  \
    using cc = cc_helper<8, 9>; __VA_ARGS__;                                  \
  } else if ((major) == 8 && (minor) == 7) {                                  \
    using cc = cc_helper<8, 7>; __VA_ARGS__;                                  \
  } else if ((major) == 8 && (minor) == 6) {                                  \
    using cc = cc_helper<8, 6>; __VA_ARGS__;                                  \
  } else if ((major) == 8 && (minor) == 0) {                                  \
    using cc = cc_helper<8, 0>; __VA_ARGS__;                                  \
  } else {                                                                    \
    using cc = cc_helper<7, 5>; __VA_ARGS__;                                  \
  }

#ifdef USE_ROCM
// Custom configuration for vectorized elementwise kernel
// with template instantiation.
namespace vectorized_templated_config {
constexpr int num_threads() {
  return 512;
}

constexpr int elems_per_thread() {
  return 32;
}

constexpr int block_work_size() {
  return elems_per_thread() * num_threads();
}
} // namespace vectorized_templated_config
#endif

template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 0) {
      return 0;
    } else {
      return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
    }
}

#ifdef USE_ROCM
template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else if constexpr (io_sizes < 4) {
    return 8;
  } else {
    return 4;
  }
}
#else
template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else {
    return 8;
  }
}
#endif


//thread work size of 8 regresses the perf of elementwise kernel on cuda
//this doesn't change ROCm behavior as thread_work_size is already 4 on ROCm
constexpr int elementwise_thread_work_size() {return 4;}
constexpr int elementwise_block_work_size() {
  return elementwise_thread_work_size() * num_threads();
}

template <int io_sizes>
constexpr auto io_block_work_size() {
  return num_threads() * elems_per_thread<io_sizes>();
}

#ifdef USE_ROCM
template <typename args_t, size_t... Is>
constexpr auto input_size(args_t args, std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) == 0) {
    return 0;
  } else {
    return sizeof(std::tuple_element_t<0, args_t>);
  }
}

template <int vec_size, int io_size>
constexpr auto calc_optimal_vec_size() {
  static_assert(vec_size != 0);
  static_assert(io_size != 0);
  if constexpr (io_size == 1 && vec_size >= 16) {
    return 16;
  } else if constexpr (io_size <= 2 && vec_size >= 8) {
    return 8;
  } else if constexpr (io_size <= 4 && vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 2) {
    return 2;
  } else {
    return 1;
  }
}
#endif

template <typename func_t>
constexpr auto calc_io_size(){
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
#ifdef USE_ROCM
  constexpr auto input_size = at::native::input_size(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return (input_size > 0) ? ((input_size < output_size) ? input_size : output_size) : output_size;
#else
  constexpr auto input_size = at::native::sum_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return input_size + output_size;
#endif
}

#ifndef USE_ROCM

struct LaunchConfig {
  dim3 grid_size, block_size;
  int shared_memory_size;
};

template <typename func_t_>
struct TuningConstantsSelector {
  using func_t = func_t_;

  template <int cc_major_, int cc_minor_, bool small_footprint_>
  struct tc {
    using func_t = func_t_;
    static constexpr int cc_major = cc_major_;
    static constexpr int cc_minor = cc_minor_;
    static constexpr bool small_footprint = small_footprint_;
    static constexpr bool is_simple =
        get_is_simple<func_t, cc_major, cc_minor>::value;
    // note: for a simple vectorized kernel, there is no point in having more than
    // 128 threads per block: this will just cause less parallelism (bad at lower sizes),
    // and using less than 128 threads per block risks under-using warp schedulers
    // on all current supported architectures. Thus, we set it to 128.
    static constexpr int threads_per_block = 128;
    // note: the bytes per thread determine how many bytes in flight the kernel
    // can achieve, along with the total number of threads per SM (not block!).
    // For sm 90, 10, 10.3, 32 bytes per thread are required, for all other
    // currently supported architectures, 16 bytes per thread is sufficient.
    // For small footprints, we always use 16 bytes per thread.
    static constexpr int bytes_per_thread = (
      ((cc_major == 8 && cc_minor == 0 && is_simple) ||
      cc_major == 9 || cc_major == 10) &&
      !small_footprint) ? 32 : 16;
    // note: the max unroll plays an important role for lambdas with many instructions:
    // if we allow too much unroll, code-generation may be far from optimal, because
    // the compiler will need to issue more and more instructions to comply with
    // register constraints. For non-simple lambdas, we thus limit unroll to 8.
    static constexpr int max_unroll = is_simple ? 32 : 8;
    // note: in order to allow more flexibility for the compiler with register
    // constraints, we allow up to 2 unroll sequences per loaded set of registers.
    static constexpr int max_vectors_per_unroll = 2;
    // for non-simple lambdas, we want to hide more latency even within a thread,
    // thus we split the load set into 2 instructions.
    static constexpr int max_bytes_per_load_inst =
        is_simple ? bytes_per_thread : bytes_per_thread / 2;
    // note: the fallback path requires significantly more registers for bounds-checking
    // and branching, so we limit the unroll more than for the vectorized path.
    static constexpr int max_unroll_fallback = is_simple ? 8 : 4;
  };
};

template <typename V = void>
struct MaxArgSize {
  static constexpr int value = 0;
};

template <typename... Ts>
struct MaxArgSize<std::tuple<Ts...>> {
  static constexpr int value = [] {
    int max_size{0};
    ((max_size = std::max(max_size, int{sizeof(Ts)})), ...);
    return max_size;
  }();
};

template <typename tc_t>
struct KernelConfig {
  using tc = tc_t;
  using func_t = typename tc::func_t;
  using func_traits = function_traits<func_t>;
  using args_tuple = typename func_traits::ArgsTuple;
  using result_type = typename func_traits::result_type;

  // this is a maximum for the load/store instruction size we must respect
  static constexpr int max_io_size = tc::cc_major >= 10 ? 32 : 16;

  // +1 for output tensor
  static constexpr int num_tensors = func_traits::arity + 1;
  static constexpr int num_inputs = func_traits::arity;

  // elements per thread depends on the max argument size and the bytes per thread
  static constexpr int max_arg_size = num_inputs == 0 ?
    int{sizeof(result_type)} : MaxArgSize<args_tuple>::value;
  static_assert(tc::bytes_per_thread % max_arg_size == 0,
    "bytes_per_thread must be a multiple of max_arg_size");
  static constexpr int elems_per_thread = tc::bytes_per_thread / max_arg_size;
  // now we know the total elems per block
  static constexpr int block_elems = elems_per_thread * tc::threads_per_block;

  // maximum number of elements that can fit in one IO instruction
  static constexpr int max_elems_per_load_inst = max_io_size / max_arg_size;
  static constexpr int max_elems_per_store_inst = max_io_size / int{sizeof(result_type)};

  static constexpr int max_total_unroll = std::min(elems_per_thread, tc::max_unroll);
  // get the store vector size from max_unroll and max io size
  static constexpr int store_vec_size = std::min(max_total_unroll, max_elems_per_store_inst);
  // get the load vector size from max_bytes_per_load_inst and max io size
  static constexpr int load_vec_size = std::min(
    ::cuda::ceil_div(tc::max_bytes_per_load_inst, max_arg_size), max_elems_per_load_inst);
  // final vec size must be the minimum of the above two
  static constexpr int vec_size_uncapped = std::min(load_vec_size, store_vec_size);
  // NVCC bug: vec_size > 4 causes numerical mismatches with 1-byte types
  // (e.g. uint8, int8) on sm80 and sm90. Cap at 4 for safety.
  static constexpr int vec_size = int{sizeof(result_type)} < 2
      ? std::min(vec_size_uncapped, 4) : vec_size_uncapped;
  // get the loads/stores per unroll according to max_vectors_per_unroll and max total unroll
  static_assert(max_total_unroll % vec_size == 0,
    "max_total_unroll must be a multiple of vec_size");
  static constexpr int vectors_per_unroll = std::min(
    max_total_unroll / vec_size, tc::max_vectors_per_unroll);

  static constexpr int elems_per_unroll = vectors_per_unroll * vec_size;
  // at this point, we can determine the number of unrolls
  static_assert(elems_per_thread % elems_per_unroll == 0,
    "elems_per_thread must be a multiple of elems_per_unroll");
  static constexpr int num_unrolls = elems_per_thread / elems_per_unroll;

  // in case we are simply unrolling, we should just limit the
  // elements per thread (and block) to max_unroll_fallback
  // note: the unrolled path typically requires significantly more registers
  // for bounds-checking and branching (required at least for every load/store),
  // so limit unroll more than for the vectorized path.
  static constexpr int elems_per_thread_unroll = std::min(elems_per_thread, tc::max_unroll_fallback);
  static constexpr int block_elems_unroll = elems_per_thread_unroll * tc::threads_per_block;

  template <typename array_t, int... Is>
  static bool inputs_same_misalignment(
      int r, array_t const& data, std::integer_sequence<int, Is...>) {
    return ((r == int(reinterpret_cast<uintptr_t>(data[Is + 1])
                 / sizeof(std::tuple_element_t<Is, args_tuple>)
                 % vec_size)) && ...);
  }

  template <typename array_t>
  static std::pair<int, int> get_aligned_offset_and_size(array_t const& data, int64_t N) {
    static_assert(std::tuple_size_v<array_t> == num_tensors, "array_t must match num_tensors");
    // note that there cannot always be a single offset and size which works
    // for all input pointers: for example, if we have two pointers with 2-byte data type sizes
    // one pointer is aligned to 4 bytes and another to 2 bytes, then there is no
    // single offset in number of elements that aligns both pointers to 4 bytes.
    // in such a case, we return a pair of -1, -1 to indicate that there is no
    // aligned offset and size that works for all pointers, and this will be handled
    // in a special way by the kernel.

    // for each tensor, compute how many elements past the last vec_size-aligned
    // boundary the pointer sits: r_i = (addr_i / sizeof(type_i)) % vec_size.
    // a common aligned_offset exists iff all r_i are equal.
    int r = int(reinterpret_cast<uintptr_t>(data[0])
                / sizeof(result_type) % vec_size);
    if (!inputs_same_misalignment(
        r, data, std::make_integer_sequence<int, num_inputs>{})) {
      return {-1, -1};
    }

    // aligned_offset is the number of elements to skip to reach the next
    // vec_size-aligned boundary (at most vec_size - 1).
    int aligned_offset = (vec_size - r) % vec_size;
    int64_t tail = N - aligned_offset;
    // round down to block_elems so that every inner block is a full tile
    int aligned_size = tail > 0
        ? static_cast<int>((tail / block_elems) * block_elems) : 0;
    return {aligned_offset, aligned_size};
  }

  static LaunchConfig launch_config(int64_t N, int aligned_size) {
    int grid_size = 0;
    if (aligned_size >= 0) {
      if constexpr (tc::small_footprint) {
        grid_size = static_cast<int>(::cuda::ceil_div(N, int64_t{block_elems}));
      } else {
        // use aligned_size (not N) so the number of inner blocks matches
        // the aligned region; the boundary blocks handle prefix/suffix.
        grid_size = static_cast<int>(
            ::cuda::ceil_div(int64_t{aligned_size}, int64_t{block_elems})) + 2;
      }
    } else {
      grid_size = static_cast<int>(::cuda::ceil_div(N, int64_t{block_elems_unroll}));
    }
    return LaunchConfig{
      dim3(grid_size, 1, 1),
      dim3(tc::threads_per_block, 1, 1),
      /*shared_memory_size*/0
    };
  }
};

template <int num_threads,
          int elems_per_thread_unroll,
          int elems_per_thread,
          typename args_tuple,
          typename result_type,
          typename func_t,
          typename array_t>
__device__ __noinline__ void vectorized_elementwise_kernel_fallback(
    int N, func_t f, array_t data, int aligned_offset, int aligned_size) {
  int block1_start = aligned_offset + aligned_size;
  // Block 0 processes [0, aligned_offset).
  // Block 1 processes [aligned_offset+aligned_size, N).
  int remaining = blockIdx.x == 0 ? aligned_offset : N - block1_start;

  // add an early exit: this avoids loading unnecessary code for common cases
  if (remaining <= 0) {
    return;
  }
  constexpr int block_elems_unroll = elems_per_thread_unroll * num_threads;
  // note: block 0 handles at most vec_size - 1 elements, so as long as
  // block_elems_unroll >= vec_size - 1, we will have a single iteration of
  // the unrolled fallback. Since elems_per_thread must be strictly greater than
  // vec_size - 1, the static_assert checks whether the above is satisfied.
  static_assert(block_elems_unroll >= elems_per_thread,
    "block_elems_unroll must be >= elems_per_thread");
  // Block 1 handles at most elems_per_thread * num_threads - 1 elements,
  // so the number of iterations is
  // ceil_div(elems_per_thread * num_threads - 1, block_elems_unroll).
  int max_iters = blockIdx.x == 0 ? 1 : ::cuda::ceil_div(
    elems_per_thread * num_threads - 1, block_elems_unroll);

  if (blockIdx.x == 1) {
    memory::policies::advance_data<result_type, args_tuple>(data, block1_start);
  }

  auto input_calc = TrivialOffsetCalculator<std::tuple_size_v<args_tuple>>();
  auto output_calc = TrivialOffsetCalculator<1>();

  #pragma unroll 1
  for (int iter = 0; iter < max_iters; ++iter) {
    auto policy = memory::policies::unroll_base<
      num_threads,
      array_t,
      decltype(input_calc),
      decltype(output_calc),
      memory::LoadWithoutCast,
      memory::StoreWithoutCast,
      elems_per_thread_unroll,
      /*num_outputs*/1,
      /*check_compute_bounds*/false>(
        data, remaining, input_calc, output_calc,
        memory::LoadWithoutCast(), memory::StoreWithoutCast());
    elementwise_kernel_helper(f, policy, /*zero_idx*/true);
    memory::policies::advance_data<result_type, args_tuple>(data, block_elems_unroll);
    remaining -= block_elems_unroll;
  }
}

// note: we explicitly put all relevant parameters as template parameters
// to facilitate profiling and debugging, as the parameters show up directly
// in the kernel name, instead of being inferred from the function signature.
// This doesn't increase the number of instantiations.
template <int num_threads,
          int elems_per_thread,
          int elems_per_thread_unroll,
          int num_unrolls,
          int vectors_per_unroll,
          int vec_size,
          bool small_footprint,
          bool is_simple,
          typename func_t,
          typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data, int aligned_offset, int aligned_size) {
  using traits = function_traits<func_t>;
  using args_tuple = typename traits::ArgsTuple;
  using result_type = typename traits::result_type;
  constexpr int block_elems = elems_per_thread * num_threads;

  if constexpr (small_footprint) {
    // for small footprints, we assume that both input/output base pointers
    // and sizes are aligned to the vector size. Thus, there is no fallback.
    // However, we still need to check the remaining elements as the size
    // might not be a multiple of the block tile size.
    memory::policies::advance_data<result_type, args_tuple>(
      data, block_elems * static_cast<int>(blockIdx.x));
    int remaining = N - block_elems * static_cast<int>(blockIdx.x);
    auto policy = memory::policies::streaming_vectorized<
        /*has_remaining*/true,
        vec_size,
        vectors_per_unroll,
        num_threads,
        array_t>(data, remaining);
    streaming_elementwise_kernel_helper<num_unrolls>(f, policy);
  } else {
    // Blocks 0 and 1 are fallback blocks handling the unaligned
    // prefix and suffix respectively. Blocks >= 2 are inner blocks
    // that process full vectorized block tiles.
    if (blockIdx.x >= 2) {
      memory::policies::advance_data<result_type, args_tuple>(data,
          aligned_offset + static_cast<int>(blockIdx.x - 2) * block_elems);
      int remaining = -1;  // inner blocks must be full tiles
      auto policy = memory::policies::streaming_vectorized<
          /*has_remaining*/false,
          vec_size,
          vectors_per_unroll,
          num_threads,
          array_t>(data, remaining);
      streaming_elementwise_kernel_helper<num_unrolls>(f, policy);
    } else {
      // note: the fallback is explicitly no-inlined, such that compiler does
      // not pollute the main path with the fallback logic.
      vectorized_elementwise_kernel_fallback<
          num_threads,
          elems_per_thread_unroll,
          elems_per_thread,
          args_tuple,
          result_type>(N, f, data, aligned_offset, aligned_size);
    }
  }
}

#else // USE_ROCM
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
#if defined(USE_ROCM) && defined(__gfx942__)
  // Similar check in launch_vectorized_kernel() as well. Both should be in sync.
  constexpr int tws = 16;
#else
  constexpr int tws = elems_per_thread<io_size>();
#endif
  constexpr int bws = tws * num_threads();
  int remaining = N - bws * blockIdx.x;

  if (remaining < bws) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast,
        tws>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    constexpr auto optimal_vec_size = calc_optimal_vec_size<vec_size, io_size>();
    elementwise_kernel_helper(
        f, memory::policies::vectorized<optimal_vec_size, array_t, tws>(data));
  }
}
#endif // USE_ROCM

template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;
  auto policy = memory::policies::
      unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t, elems_per_thread>(
          data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
#ifdef USE_ROCM
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  c10::DeviceIndex curDevice = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&curDevice));
  // Similar check in vectorized_elementwise_kernel() as well. Both should be in sync.
  int tws = at::detail::getCUDAHooks().isGPUArch({"gfx942"}, curDevice) ? 16 : elems_per_thread<io_size>();
  int bws = tws * num_threads();
  int64_t grid = (N + bws - 1) / bws;
  switch (vec_size) {
    case 16:
      vectorized_elementwise_kernel<16, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 8:
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      int64_t grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
          <<<grid_unrolled, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
#else // USE_ROCM
  // here we determine whether we have a small footprint or not
  // by default, we don't have a small footprint
  bool use_small_footprint = false;
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  AT_DISPATCH_CC(p->major, p->minor, [&] {
    using tc_selector_t = TuningConstantsSelector<func_t>;
    using tc_default = typename tc_selector_t::template tc<cc::major, cc::minor, false>;
    using tc_small = typename tc_selector_t::template tc<cc::major, cc::minor, true>;
    constexpr bool is_simple = get_is_simple<func_t, cc::major, cc::minor>::value;
    // there is no way to obtain the number of instructions of a kernel, but
    // here we are making a guess / heuristic which states that a simple function
    // has ~200 instructions while a non-simple function has ~1000 instructions,
    // for the default kernel path (no small footprint / with fallback), with
    // a total unroll of 4.
    constexpr int default_n_instructions_unroll4 = is_simple ? 200 : 1000;
    // determine the actual unroll size we would have
    using config_default = KernelConfig<tc_default>;
    constexpr int default_unroll = config_default::elems_per_thread;
    constexpr int default_n_instructions = default_unroll >= 4 ?
      (default_n_instructions_unroll4 * default_unroll) / 4 :
      (default_n_instructions_unroll4 * 4) / default_unroll;
    constexpr int default_size = default_n_instructions * 16;
    // note : each SM needs to load the code once, at the very least
    int64_t default_code_footprint = p->multiProcessorCount * default_size;
    // we determine how many bytes per element we need to load
    using args_t = typename function_traits<func_t>::ArgsTuple;
    constexpr int bytes_per_element = at::native::sum_of_sizes(
      args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
    // the expected load footprint for data
    int64_t data_footprint = N * int64_t{bytes_per_element};
    // if the code footprint isn't smaller than the data footprint,
    // we consider the input size to be small
    bool is_small_footprint = default_code_footprint >= data_footprint;
    // just because we have a small input size doesn't mean we can launch
    // the small footprint kernel, as it requires alignment (no fallback).
    // thus, we determine whether we are aligned
    using config_small = KernelConfig<tc_small>;
    if (is_small_footprint) {
      // if we can vectorize up to the required vector size and the number of elements
      // aligns with the required vector size, we can use the small footprint kernel
      if (memory::can_vectorize_up_to<func_t>(data) >= config_small::vec_size &&
          (N % config_small::vec_size == 0)) {
        use_small_footprint = true;
      }
    }
    if (use_small_footprint) {
      // note aligned offset and size must not be used by the small footprint kernel
      int aligned_offset = 0, aligned_size = 0;
      auto lc = config_small::launch_config(N, aligned_size);
      auto& kernel = vectorized_elementwise_kernel<
        tc_small::threads_per_block,
        config_small::elems_per_thread,
        config_small::elems_per_thread_unroll,
        config_small::num_unrolls,
        config_small::vectors_per_unroll,
        config_small::vec_size,
        /*small_footprint*/true,
        tc_small::is_simple,
        func_t,
        array_t>;
      kernel<<<lc.grid_size, lc.block_size, lc.shared_memory_size, stream>>>(
          N, f, data, aligned_offset, aligned_size);
      return;
    }
    auto [aligned_offset, aligned_size] = config_default::get_aligned_offset_and_size(data, N);
    if (aligned_offset < 0) {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      int64_t grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
          <<<grid_unrolled, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      return;
    }
    TORCH_INTERNAL_ASSERT(aligned_offset >= 0);
    TORCH_INTERNAL_ASSERT(aligned_offset < config_default::vec_size);
    TORCH_INTERNAL_ASSERT(aligned_size >= 0 && aligned_size <= N);
    TORCH_INTERNAL_ASSERT(N - aligned_offset - aligned_size < config_default::block_elems);
    auto lc = config_default::launch_config(N, aligned_size);
    auto& kernel = vectorized_elementwise_kernel<
      tc_default::threads_per_block,
      config_default::elems_per_thread,
      config_default::elems_per_thread_unroll,
      config_default::num_unrolls,
      config_default::vectors_per_unroll,
      config_default::vec_size,
      /*small_footprint*/false,
      tc_default::is_simple,
      func_t,
      array_t>;
    kernel<<<lc.grid_size, lc.block_size, lc.shared_memory_size, stream>>>(
        N, f, data, aligned_offset, aligned_size);
  }());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif // USE_ROCM
}

#ifdef USE_ROCM
template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
C10_LAUNCH_BOUNDS_1(vectorized_templated_config::num_threads())
__global__ void vectorized_templated_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t inp_calc,
    out_calc_t out_calc,
    loader_t loader,
    storer_t storer) {
  int remaining = N -
      vectorized_templated_config::block_work_size() *
          (gridDim.x - blockIdx.x - 1);
  constexpr bool reverted_idx = true;

  if (remaining <
      vectorized_templated_config::block_work_size()) { // if this block handles
                                                        // the reminder,
    // just do a naive unrolled loop
    auto policy = memory::policies::unroll_base<
        vectorized_templated_config::num_threads(),
        array_t,
        inp_calc_t,
        out_calc_t,
        loader_t,
        storer_t,
        vectorized_templated_config::elems_per_thread()>(
        data, remaining, inp_calc, out_calc, loader, storer);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    auto policy = memory::policies::vectorized_templated<
        vec_size,
        array_t,
        vectorized_templated_config::elems_per_thread(),
        vectorized_templated_config::num_threads(),
        OutputType,
        InputTypes...>(data);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  }
}

// This function assume trivial 1d and supports template specialization
// to avoid dynamic casting.
// Input vectorization size is based on runtime information, i.e.
// the actual data types of the input and output tensor and cannot
// be determined using the functor type, as in regular non-templated
// vectorized kernels. The caller is in charge of selecting the correct input
// vectorization length.
template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
static inline void launch_vectorized_templated_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + vectorized_templated_config::block_work_size() - 1) /
      vectorized_templated_config::block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  switch (vec_size) {
    case 8:
      vectorized_templated_elementwise_kernel<
          8,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_templated_elementwise_kernel<
          4,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_templated_elementwise_kernel<
          2,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    default:
      // vector size 1 is not handled as part of vectorize_templated kernel
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
#endif

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int64_t grid = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename func_t>
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

template <int nt, int vt, typename func_t>
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

#ifdef USE_ROCM
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_manual_unroll(int N, func_t f) {
  int tid = threadIdx.x;
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  if ((idx + nt*(vt-1)) < N) {
    f(idx, true);
  } else {
#pragma unroll
    for (int i = 0; i < vt; i++) {
      if (idx < N) {
        f(idx, false);
        idx += nt;
      }
    }
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel_manual_unroll(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_manual_unroll<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
#ifndef USE_ROCM
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data[1], &offsets[1], 1);
  });
#else
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 4 : 8;
  constexpr int grp_sz = 128;
  launch_legacy_kernel_manual_unroll<grp_sz, unroll_factor>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
    if (unrl) {
      if constexpr (unroll_factor == 4) {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx+grp_sz);
        auto offsets2 = offset_calc.get(idx+grp_sz*2);
        auto offsets3 = offset_calc.get(idx+grp_sz*3);
        arg0_t* out0 = (arg0_t*)(data[0] + offsets0[0]);
        arg0_t* out1 = (arg0_t*)(data[0] + offsets1[0]);
        arg0_t* out2 = (arg0_t*)(data[0] + offsets2[0]);
        arg0_t* out3 = (arg0_t*)(data[0] + offsets3[0]);
        auto tmp0 = invoke(f, &data[1], &offsets0[1], 1);
        auto tmp1 = invoke(f, &data[1], &offsets1[1], 1);
        auto tmp2 = invoke(f, &data[1], &offsets2[1], 1);
        auto tmp3 = invoke(f, &data[1], &offsets3[1], 1);
        *out0 = tmp0;
        *out1 = tmp1;
        *out2 = tmp2;
        *out3 = tmp3;
      } else {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx+grp_sz);
        auto offsets2 = offset_calc.get(idx+grp_sz*2);
        auto offsets3 = offset_calc.get(idx+grp_sz*3);
        auto offsets4 = offset_calc.get(idx+grp_sz*4);
        auto offsets5 = offset_calc.get(idx+grp_sz*5);
        auto offsets6 = offset_calc.get(idx+grp_sz*6);
        auto offsets7 = offset_calc.get(idx+grp_sz*7);
        arg0_t* out0 = (arg0_t*)(data[0] + offsets0[0]);
        arg0_t* out1 = (arg0_t*)(data[0] + offsets1[0]);
        arg0_t* out2 = (arg0_t*)(data[0] + offsets2[0]);
        arg0_t* out3 = (arg0_t*)(data[0] + offsets3[0]);
        arg0_t* out4 = (arg0_t*)(data[0] + offsets4[0]);
        arg0_t* out5 = (arg0_t*)(data[0] + offsets5[0]);
        arg0_t* out6 = (arg0_t*)(data[0] + offsets6[0]);
        arg0_t* out7 = (arg0_t*)(data[0] + offsets7[0]);
        auto tmp0 = invoke(f, &data[1], &offsets0[1], 1);
        auto tmp1 = invoke(f, &data[1], &offsets1[1], 1);
        auto tmp2 = invoke(f, &data[1], &offsets2[1], 1);
        auto tmp3 = invoke(f, &data[1], &offsets3[1], 1);
        auto tmp4 = invoke(f, &data[1], &offsets4[1], 1);
        auto tmp5 = invoke(f, &data[1], &offsets5[1], 1);
        auto tmp6 = invoke(f, &data[1], &offsets6[1], 1);
        auto tmp7 = invoke(f, &data[1], &offsets7[1], 1);
        *out0 = tmp0;
        *out1 = tmp1;
        *out2 = tmp2;
        *out3 = tmp3;
        *out4 = tmp4;
        *out5 = tmp5;
        *out6 = tmp6;
        *out7 = tmp7;
      }
    } else {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      *out = invoke(f, &data[1], &offsets[1], 1);
    }
  });
#endif
}

#ifdef USE_ROCM
namespace {
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity,
    size_t arg_num = 0>
struct check_binary_functor_types_for_specialization {
  constexpr static inline bool check() {
    if constexpr (arity != 2)
      return false;
    if constexpr (arg_num == 0) {
      using SelectedType = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<FirstParamTy, SelectedType>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    } else if constexpr (arg_num == 1) {
      using SelectedType2 = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<SecondParamTy, SelectedType2>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    }
    return false;
  }
};

// Bottom case: if we got this far, assume correct type matching except
// when there are no arguments (arity == 0).
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    arity,
    arity> {
  constexpr static inline bool check() {
    if constexpr (arity != 0)
      return true;
    return false;
  }
};

template <typename TupleLike, typename FirstParamTy, typename SecondParamTy>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    0,
    0> {
  constexpr static inline bool check() {
    return false;
  }
};

// The following is a list of type specializations for vectorized_templated
// elementwise kernel. The three types refer to runtime types of the output
// tensor, first tensor argument, and the second tensor argument used for a
// binary functor.
constexpr std::array rt_binary_specializations = {
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value})};

bool check_binary_rt_types_for_specialization(TensorIteratorBase& iter) {
  if (iter.ninputs() != 2)
    return false;
  for (auto spec : rt_binary_specializations)
    if (iter.dtype(0) == spec[0] && iter.input_dtype(0) == spec[1] &&
        iter.input_dtype(1) == spec[2])
      return true;
  return false;
}

template <int arg_index>
struct type_specialized_kernel_launcher {
  template <
      typename func_t,
      typename array_t,
      typename inp_calc_t,
      typename out_calc_t,
      typename loader_t,
      typename storer_t>
  static void apply(
      ScalarType ret_t,
      ScalarType arg0_t,
      ScalarType arg1_t,
      int64_t numel,
      func_t f,
      array_t data,
      inp_calc_t input_offset_calculator,
      out_calc_t output_offset_calculator,
      loader_t loader,
      storer_t storer) {
    constexpr ScalarType sret_t = rt_binary_specializations[arg_index][0];
    constexpr ScalarType sarg0_t = rt_binary_specializations[arg_index][1];
    constexpr ScalarType sarg1_t = rt_binary_specializations[arg_index][2];
    if (ret_t == sret_t && arg0_t == sarg0_t && arg1_t == sarg1_t) {
      using cret_t = c10::impl::ScalarTypeToCPPTypeT<sret_t>;
      using carg0_t = c10::impl::ScalarTypeToCPPTypeT<sarg0_t>;
      using carg1_t = c10::impl::ScalarTypeToCPPTypeT<sarg1_t>;
      launch_vectorized_templated_kernel<
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          cret_t,
          carg0_t,
          carg1_t>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    }
  }
};

template <int arg_index>
struct type_specialized_broadcast_kernel_launcher {
  template <
      typename func_t,
      typename array_t,
      typename dtypes_t,
      typename calc_t>
  static void apply(
      int64_t numel,
      func_t f,
      array_t data,
      dtypes_t dtypes,
      calc_t offset_calc) {
        using traits = function_traits<func_t>;
        using ret_t = typename traits::result_type;
        using arg0_t = typename traits::template arg<0>::type;
        using arg1_t = typename traits::template arg<1>::type;
        if (dtypes[0] == rt_binary_specializations[arg_index][0] &&
          dtypes[1] == rt_binary_specializations[arg_index][1] &&
          dtypes[2] == rt_binary_specializations[arg_index][2]) {
            using ret_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][0]>;
            using arg0_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][1]>;
            using arg1_cpp_t = c10::impl::ScalarTypeToCPPTypeT<rt_binary_specializations[arg_index][2]>;
            constexpr int grp_sz = 128;
            launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
              if (unrl) {
                auto offsets0 = offset_calc.get(idx);
                auto offsets1 = offset_calc.get(idx + grp_sz);
                auto offsets2 = offset_calc.get(idx + grp_sz * 2);
                auto offsets3 = offset_calc.get(idx + grp_sz * 3);
                void* out0 = data[0] + offsets0[0];
                void* out1 = data[0] + offsets1[0];
                void* out2 = data[0] + offsets2[0];
                void* out3 = data[0] + offsets3[0];
                auto u = c10::load<arg0_cpp_t>(data[1] + offsets0[1]);
                auto v = c10::load<arg1_cpp_t>(data[2] + offsets0[2]);
                ret_t result0 = f(c10::convert<arg0_t>(u), c10::convert<arg1_t>(v));
                auto u1 = c10::load<arg0_cpp_t>(data[1] + offsets1[1]);
                auto v1 = c10::load<arg1_cpp_t>(data[2]+ offsets1[2]);
                ret_t result1 = f(c10::convert<arg0_t>(u1), c10::convert<arg1_t>(v1));
                auto u2 = c10::load<arg0_cpp_t>(data[1] + offsets2[1]);
                auto v2 = c10::load<arg1_cpp_t>(data[2] + offsets2[2]);
                ret_t result2 = f(c10::convert<arg0_t>(u2), c10::convert<arg1_t>(v2));
                auto u3 = c10::load<arg0_cpp_t>(data[1] + offsets3[1]);
                auto v3 = c10::load<arg1_cpp_t>(data[2] + offsets3[2]);
                ret_t result3 = f(c10::convert<arg0_t>(u3), c10::convert<arg1_t>(v3));
                *(ret_cpp_t*)out0 = c10::convert<ret_cpp_t>(result0);
                *(ret_cpp_t*)out1 = c10::convert<ret_cpp_t>(result1);
                *(ret_cpp_t*)out2 = c10::convert<ret_cpp_t>(result2);
                *(ret_cpp_t*)out3 = c10::convert<ret_cpp_t>(result3);
              } else {
                auto offsets = offset_calc.get(idx);
                void* out = data[0] + offsets[0];
                auto u = c10::load<arg0_cpp_t>(data[1] + offsets[1]);
                auto v = c10::load<arg1_cpp_t>(data[2] + offsets[2]);
                ret_t result = f(c10::convert<arg0_t>(u), c10::convert<arg1_t>(v));
                *(ret_cpp_t*)out = c10::convert<ret_cpp_t>(result);
              }
            });
        }
      }
};

} // namespace
#endif

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

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
#ifdef USE_ROCM
    // Attempt to call specialized vectorized elementwise kernel
    // that enables interleaving.
    if (check_binary_rt_types_for_specialization(iter) &&
        memory::can_vectorize_up_to<func_t>(data) > 1) {
      // constexpr to reduce the amount of kernels generated for
      // vectorized templated elementwise and limit which functors are actually
      // applied to the load and store at compile time.
      using func_tuple = typename traits::ArgsTuple;
      if constexpr (
          std::is_same_v<float, arg0_t> && traits::arity == 2 &&
          check_binary_functor_types_for_specialization<
              func_tuple,
              float,
              float,
              traits::arity,
              /*arg_num=*/0>::check()) {
        // If we got here, we know we are in one of the specialized cases. We
        // need to translate the runtime type to a statically known type. This
        // is effectively hoisting to the host the switch over runtime type in
        // the kernel in fetch_and_cast. Loader, storer, offset calculators are
        // only needed for the reminder loop.
        auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
        auto output_offset_calculator = TrivialOffsetCalculator<1>();
        auto loader = memory::LoadWithCast<traits::arity>(iter);
        auto storer = memory::StoreWithCast<1>(iter);
        memory::detail::static_unroll<
            type_specialized_kernel_launcher,
            rt_binary_specializations.size()>::
            with_args(
                iter.dtype(0),
                iter.input_dtype(0),
                iter.input_dtype(1),
                numel,
                f,
                data,
                input_offset_calculator,
                output_offset_calculator,
                loader,
                storer);
        return;
      }
    }
    std::array<ScalarType, ntensors> dtypes;
    auto inner_strides = iter.get_inner_strides();
    std::array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
      strides[i] = inner_strides[i];
    }
    constexpr int grp_sz = 128;
    launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
      if (unrl) {
        void* out0 = data[0] + strides[0] * idx;
        void* out1 = data[0] + strides[0] * (idx + grp_sz);
        void* out2 = data[0] + strides[0] * (idx + grp_sz * 2);
        void* out3 = data[0] + strides[0] * (idx + grp_sz * 3);
        arg0_t result0 = invoke(f, &data[1], &strides[1], &dtypes[1], idx);
        arg0_t result1 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz));
        arg0_t result2 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz * 2));
        arg0_t result3 = invoke(f, &data[1], &strides[1], &dtypes[1], (idx + grp_sz * 3));
        c10::cast_and_store<arg0_t>(dtypes[0], out0, result0);
        c10::cast_and_store<arg0_t>(dtypes[0], out1, result1);
        c10::cast_and_store<arg0_t>(dtypes[0], out2, result2);
        c10::cast_and_store<arg0_t>(dtypes[0], out3, result3);
      } else {
        void* out = data[0] + strides[0] * idx;
        arg0_t result = invoke(f, &data[1], &strides[1], &dtypes[1], idx);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      }
    });
#else
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
#endif
  } else {
    std::array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
#ifdef USE_ROCM
    if (check_binary_rt_types_for_specialization(iter)) {
      // constexpr to reduce the amount of kernels generated for
      // broadcast elementwise with mexed dtypes and limit which functors are actually
      // applied to the load and store at compile time.
      using func_tuple = typename traits::ArgsTuple;
      if constexpr (
        std::is_same_v<float, arg0_t> && traits::arity == 2 &&
        check_binary_functor_types_for_specialization<
          func_tuple,
          float,
          float,
          traits::arity,
          /*arg_num=*/0>::check()) {
            memory::detail::static_unroll<
              type_specialized_broadcast_kernel_launcher,
              rt_binary_specializations.size()>::with_args(
                numel,
                f,
                data,
                dtypes,
                offset_calc
            );
            return;
      }
    }

    constexpr int grp_sz = 128;
    launch_legacy_kernel_manual_unroll<grp_sz, 4>(numel, [=] GPU_LAMBDA(int idx, bool unrl) {
      if (unrl) {
        auto offsets0 = offset_calc.get(idx);
        auto offsets1 = offset_calc.get(idx + grp_sz);
        auto offsets2 = offset_calc.get(idx + grp_sz * 2);
        auto offsets3 = offset_calc.get(idx + grp_sz * 3);
        void* out0 = data[0] + offsets0[0];
        void* out1 = data[0] + offsets1[0];
        void* out2 = data[0] + offsets2[0];
        void* out3 = data[0] + offsets3[0];
        arg0_t result0 = invoke(f, &data[1], &offsets0[1], &dtypes[1], 1);
        arg0_t result1 = invoke(f, &data[1], &offsets1[1], &dtypes[1], 1);
        arg0_t result2 = invoke(f, &data[1], &offsets2[1], &dtypes[1], 1);
        arg0_t result3 = invoke(f, &data[1], &offsets3[1], &dtypes[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out0, result0);
        c10::cast_and_store<arg0_t>(dtypes[0], out1, result1);
        c10::cast_and_store<arg0_t>(dtypes[0], out2, result2);
        c10::cast_and_store<arg0_t>(dtypes[0], out3, result3);
      } else {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      }
    });
#else
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
#endif
  }
}

#undef AT_DISPATCH_CC

} // namespace at::native
