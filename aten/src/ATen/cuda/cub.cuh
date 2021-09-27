#pragma once

#include <cstddef>
#include <type_traits>
#include <iterator>
#include <limits>

// include cub in a safe manner, see:
// https://github.com/pytorch/pytorch/pull/55292
#undef CUB_NS_POSTFIX //undef to avoid redefinition warnings
#undef CUB_NS_PREFIX
#define CUB_NS_PREFIX namespace at { namespace cuda { namespace detail {
#define CUB_NS_POSTFIX }}}
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX

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

#ifdef __HIP_PLATFORM_HCC__
#define NO_ROCM(x)
#else
#define NO_ROCM(x) x

namespace at { namespace native {

namespace cub = at::cuda::detail::cub;

}}
#endif

namespace at {
namespace cuda {

namespace detail {

template<typename T>
struct cuda_type {
  using type = T;
};
template<>
struct cuda_type<c10::Half> {
  using type = __half;
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11050
// cub sort support for __nv_bfloat16 is added to cub 1.13 in
// https://github.com/NVIDIA/cub/pull/306 and according to
// https://github.com/NVIDIA/cub#releases, 1.13 is included in
// CUDA Toolkit 11.5

// waiting for https://github.com/NVIDIA/cub/pull/306 to land on CUDA
template<>
struct cuda_type<c10::BFloat16> {
  using type = __nv_bfloat16;
};

#elif !defined(__HIP_PLATFORM_HCC__)

// backport https://github.com/NVIDIA/cub/pull/306 for c10::BFloat16

template <>
struct cub::FpLimits<c10::BFloat16>
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

template <> struct cub::NumericTraits<c10::BFloat16>: cub::BaseTraits<cub::FLOATING_POINT, true, false, unsigned short, c10::BFloat16> {};

#endif

}  // namespace detail

namespace cub {

inline int get_num_bits(uint64_t max_key) {
  int num_bits = 1;
  while (max_key > 1) {
    max_key >>= 1;
    num_bits++;
  }
  return num_bits;
}

template<typename key_t>
static inline void sort_keys(
    const key_t *keys_in, key_t *keys_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortKeysDescending,
      keys_in_, keys_out_, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortKeys,
      keys_in_, keys_out_, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template<typename key_t, typename value_t>
static inline void sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in, values_out, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairs,
      keys_in_, keys_out_, values_in, values_out, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template<typename key_t, typename value_t, typename OffsetIteratorT>
static inline void segmented_sort_pairs(
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
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceSegmentedRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceSegmentedRadixSort::SortPairs,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

namespace impl {

template<typename InputIteratorT1, typename InputIteratorT2, typename OutputIteratorT, class ScanOpT>
C10_LAUNCH_BOUNDS_1(1)
__global__ void transform_vals(InputIteratorT1 a, InputIteratorT2 b, OutputIteratorT out, ScanOpT scan_op){
  *out = scan_op(*a, *b);
}

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

}

template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
inline void inclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::InclusiveScan,
      input,
      output,
      scan_op,
      size_cub,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using input_t = std::remove_reference_t<decltype(*input)>;
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
    using ArgIndexInputIterator = NO_ROCM(detail)::cub::ArgIndexInputIterator<InputIteratorT>;
    using tuple = typename ArgIndexInputIterator::value_type;
    auto input_iter_transform = [=] __device__ (const tuple &x)->input_t  {
      if (x.key == 0) {
        return *first_elem_ptr;
      } else {
        return x.value;
      }
    };
    auto input_ = NO_ROCM(detail)::cub::TransformInputIterator<input_t, decltype(input_iter_transform), ArgIndexInputIterator>(
      ArgIndexInputIterator(input + i), input_iter_transform);
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::InclusiveScan,
        input_,
        output + i,
        scan_op,
        size_cub,
        at::cuda::getCurrentCUDAStream());
  }
}

template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT>
inline void exclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, InitValueT init_value, int64_t num_items) {
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::ExclusiveScan,
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
    auto input_ = impl::chained_iterator<InitValueT, InputIteratorT>{
      input + i, first_elem_ptr};
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::InclusiveScan,
        input_,
        output + i,
        scan_op,
        size_cub,
        at::cuda::getCurrentCUDAStream());
  }
}

template<typename InputIteratorT , typename OutputIteratorT , typename NumSelectedIteratorT >
inline void unique(InputIteratorT input, OutputIteratorT output, NumSelectedIteratorT num_selected_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
    "cub unique does not support more than INT_MAX elements");
  CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceSelect::Unique,
    input, output, num_selected_out, num_items, at::cuda::getCurrentCUDAStream());
}

}}}  // namespace at::cuda::cub
