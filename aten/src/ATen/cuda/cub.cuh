#pragma once
#include <ATen/cuda/cub.h>

#include <cstddef>
#include <type_traits>
#include <iterator>
#include <limits>

#include <ATen/cuda/cub_definitions.cuh>

#if USE_GLOBAL_CUB_WRAPPED_NAMESPACE()

#include <cub/cub.cuh>

#else

// include cub in a safe manner, see:
// https://github.com/pytorch/pytorch/pull/55292
#undef CUB_NS_POSTFIX //undef to avoid redefinition warnings
#undef CUB_NS_PREFIX
#undef CUB_NS_QUALIFIER
#define CUB_NS_PREFIX namespace at_cuda_detail {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::at_cuda_detail::cub
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#undef CUB_NS_QUALIFIER

#endif

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...) do {                                       \
  size_t temp_storage_bytes = 0;                                          \
  func(nullptr, temp_storage_bytes, __VA_ARGS__);                         \
  auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();    \
  auto temp_storage = caching_allocator.allocate(temp_storage_bytes);     \
  func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);              \
  AT_CUDA_CHECK(cudaGetLastError());                                      \
} while (false)

#ifdef USE_ROCM
#define NO_ROCM(x)
#define ROCM_HIPCUB(x) ::hipcub
#else
#define NO_ROCM(x) x
#define ROCM_HIPCUB(x) x
#endif

#if (!defined(USE_ROCM) && !CUB_SUPPORTS_NV_BFLOAT16()) || defined(USE_ROCM)

#if !defined(USE_ROCM)
namespace at_cuda_detail {
#endif

// backport https://github.com/NVIDIA/cub/pull/306 for c10::BFloat16

template <>
struct ROCM_HIPCUB(cub)::FpLimits<c10::BFloat16>
{
    static __host__ __device__ __forceinline__ c10::BFloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<c10::BFloat16&>(max_word);
    }

    static __host__ __device__ __forceinline__ c10::BFloat16 Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<c10::BFloat16&>(lowest_word);
    }
};

template <>
struct ROCM_HIPCUB(cub)::NumericTraits<c10::BFloat16>:
       ROCM_HIPCUB(cub)::BaseTraits<ROCM_HIPCUB(cub)::FLOATING_POINT, true, false, unsigned short, c10::BFloat16> {};

#if !defined(USE_ROCM)
} // namespace at_cuda_detail
#endif

#endif

#if !defined(USE_ROCM)
namespace at::native {
namespace cub = ::at_cuda_detail::cub;
} // namespace at::native
#endif

namespace at::cuda::cub {

namespace detail {

template<typename T>
struct cuda_type {
  using type = T;
};
template<>
struct cuda_type<c10::Half> {
  using type = __half;
};

#if !defined(USE_ROCM) && CUB_SUPPORTS_NV_BFLOAT16()

template<>
struct cuda_type<c10::BFloat16> {
  using type = __nv_bfloat16;
};

#elif defined(USE_ROCM)

template<>
struct cuda_type<c10::BFloat16> {
  using type = hip_bfloat16;
};

#endif

}  // namespace detail

template<typename key_t, typename value_t, typename OffsetIteratorT>
inline void segmented_sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t num_elements, int64_t num_segments,
    OffsetIteratorT begin_offsets, OffsetIteratorT end_offsets,
    bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  TORCH_CHECK(num_elements <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  TORCH_CHECK(num_segments <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(num_elements * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairs,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

#if CUB_SUPPORTS_UNIQUE_BY_KEY()
template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename KeysOutputIteratorT, typename ValuesOutputIteratorT, typename NumSelectedIteratorT>
inline void unique_by_key(
  KeysInputIteratorT keys_in, ValuesInputIteratorT values_in,
  KeysOutputIteratorT keys_out, ValuesOutputIteratorT values_out,
  NumSelectedIteratorT num_selected, int64_t num_input_items)
{
  // TODO: use thrust::discard_iterator to handle null keys_out when https://github.com/NVIDIA/cub/issues/406 is fixed.
  constexpr bool null_keys_out = std::is_same<KeysOutputIteratorT, std::nullptr_t>::value;
  using KeyT = typename std::iterator_traits<KeysInputIteratorT>::value_type;
  using RealKeysOutputIteratorT = typename std::conditional<null_keys_out, KeyT *, KeysOutputIteratorT>::type;
  RealKeysOutputIteratorT keys_out_;
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;
  if constexpr (null_keys_out) {
    keys_out_owner = allocator->allocate(num_input_items * sizeof(KeyT));
    keys_out_ = static_cast<KeyT *>(keys_out_owner.get());
  } else {
    keys_out_ = keys_out;
  }
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSelect::UniqueByKey,
    keys_in, values_in, keys_out_, values_out, num_selected, num_input_items, c10::cuda::getCurrentCUDAStream());
}
#endif

namespace impl {

template<typename InputIteratorT1, typename InputIteratorT2, typename OutputIteratorT, class ScanOpT>
C10_LAUNCH_BOUNDS_1(1)
__global__ void transform_vals(InputIteratorT1 a, InputIteratorT2 b, OutputIteratorT out, ScanOpT scan_op){
  // NOTE: out here not the final scan output, but an intermediate of the accumulation type.
  using acc_t = typename std::iterator_traits<OutputIteratorT>::value_type;
  *out = scan_op(static_cast<acc_t>(*a), static_cast<acc_t>(*b));
}

#if !CUB_SUPPORTS_FUTURE_VALUE()
template<typename ValueT, typename InputIteratorT>
struct chained_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = ValueT;
  using pointer           = ValueT*;
  using reference         = ValueT&;

  InputIteratorT iter;
  ValueT *first;
  difference_type offset = 0;

  __device__ ValueT operator[](difference_type i) {
    i +=  offset;
    if (i == 0) {
      return *first;
    } else {
      return ValueT(iter[i - 1]);
    }
  }
  __device__ chained_iterator operator+(difference_type i) {
    return chained_iterator{iter, first, i};
  }
  __device__ ValueT operator*() {
    return (*this)[0];
  }
};
#endif

// even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
// so split at int_max/2
constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
}

// non synchronizing cub call
// even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
// so split at int_max/2
template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, int max_cub_size=impl::max_cub_size>
inline void inclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
#if defined(USE_ROCM)
  //For ROCm, use hipCUB chained iterators
  CUB_WRAPPER(NO_ROCM(detail)::hipcub::DeviceScan::InclusiveScan,
      input,
      output,
      scan_op,
      num_items,
      at::cuda::getCurrentCUDAStream());
  C10_HIP_KERNEL_LAUNCH_CHECK();
#else
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::InclusiveScan,
      input,
      output,
      scan_op,
      size_cub,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using input_t = typename std::iterator_traits<InputIteratorT>::value_type;
  for (int64_t i = max_cub_size; i < num_items; i += max_cub_size) {
    auto allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr first_elem = allocator->allocate(sizeof(input_t));
    auto first_elem_ptr = reinterpret_cast<input_t *>(first_elem.get());

    size_cub = std::min<int64_t>(num_items - i, max_cub_size);
    impl::transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        output + i - 1,
        input + i,
        first_elem_ptr,
        scan_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#if !CUB_SUPPORTS_FUTURE_VALUE()
    using ArgIndexInputIterator = NO_ROCM(at_cuda_detail)::cub::ArgIndexInputIterator<InputIteratorT>;
    using tuple = typename ArgIndexInputIterator::value_type;
    auto input_iter_transform = [=] __device__ (const tuple &x)->input_t  {
      if (x.key == 0) {
        return *first_elem_ptr;
      } else {
        return x.value;
      }
    };
    auto input_ = NO_ROCM(at_cuda_detail)::cub::TransformInputIterator<input_t, decltype(input_iter_transform), ArgIndexInputIterator>(
      ArgIndexInputIterator(input + i), input_iter_transform);
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::InclusiveScan,
        input_,
        output + i,
        scan_op,
        size_cub,
        at::cuda::getCurrentCUDAStream());
#else
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
        input + i + 1,
        output + i,
        scan_op,
        ::at_cuda_detail::cub::FutureValue<input_t>(first_elem_ptr),
        size_cub,
        at::cuda::getCurrentCUDAStream());
#endif
  }
#endif
}

template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, int max_cub_size=impl::max_cub_size>
inline void exclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, InitValueT init_value, int64_t num_items) {
#if defined(USE_ROCM)
  //For ROCm, use hipCUB chained iterators
  CUB_WRAPPER(NO_ROCM(detail)::hipcub::DeviceScan::ExclusiveScan,
      input,
      output,
      scan_op,
      init_value,
      num_items,
      at::cuda::getCurrentCUDAStream());
  C10_HIP_KERNEL_LAUNCH_CHECK();
#else
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
      input,
      output,
      scan_op,
      init_value,
      size_cub,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int64_t i = max_cub_size; i < num_items; i += max_cub_size) {
    auto allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr first_elem = allocator->allocate(sizeof(InitValueT));
    auto first_elem_ptr = reinterpret_cast<InitValueT *>(first_elem.get());

    size_cub = std::min<int64_t>(num_items - i, max_cub_size);
    impl::transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        output + i - 1,
        input + i - 1,
        first_elem_ptr,
        scan_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#if !CUB_SUPPORTS_FUTURE_VALUE()
    auto input_ = impl::chained_iterator<InitValueT, InputIteratorT>{
      input + i, first_elem_ptr};
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::InclusiveScan,
        input_,
        output + i,
        scan_op,
        size_cub,
        at::cuda::getCurrentCUDAStream());
#else
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
        input + i,
        output + i,
        scan_op,
        ::at_cuda_detail::cub::FutureValue<InitValueT>(first_elem_ptr),
        size_cub,
        at::cuda::getCurrentCUDAStream());
#endif
  }
#endif
}

#if CUB_SUPPORTS_SCAN_BY_KEY()

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void inclusive_sum_by_key(KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
    "cub InclusiveSumByKey does not support more than INT_MAX elements");
  CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveSumByKey,
      keys, input, output, num_items, at_cuda_detail::cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT>
inline void inclusive_scan_by_key(KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
    "cub InclusiveSumByKey does not support more than INT_MAX elements");
  CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveScanByKey,
      keys, input, output, scan_op, num_items, at_cuda_detail::cub::Equality(), at::cuda::getCurrentCUDAStream());
}

#endif

template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT>
void unique(InputIteratorT input, OutputIteratorT output,
            NumSelectedIteratorT num_selected_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub unique does not support more than INT_MAX elements");
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSelect::Unique,
              input, output, num_selected_out, num_items, at::cuda::getCurrentCUDAStream());
}

template <typename InputIteratorT, typename OutputIteratorT, typename CountsOutputIteratorT,
          typename LengthOutputIteratorT>
void run_length_encode(InputIteratorT input, OutputIteratorT output, CountsOutputIteratorT counts_out,
                       LengthOutputIteratorT length_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub run_length_encode does not support more than INT_MAX elements");
  CUB_WRAPPER(
      NO_ROCM(at_cuda_detail)::cub::DeviceRunLengthEncode::Encode,
      input, output, counts_out, length_out, num_items,
      at::cuda::getCurrentCUDAStream());
}

template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
void reduce(InputIteratorT input, OutputIteratorT output, int64_t num_items, ReductionOpT op, T init) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub reduce does not support more than INT_MAX elements");
  CUB_WRAPPER(
      NO_ROCM(at_cuda_detail)::cub::DeviceReduce::Reduce,
      input, output, num_items, op, init,
      at::cuda::getCurrentCUDAStream());

}

}  // namespace at::cuda::cub
