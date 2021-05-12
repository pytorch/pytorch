#pragma once

#include <cstddef>

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
namespace cub {

template<typename T>
struct cuda_type {
  using type = T;
};
template<>
struct cuda_type<c10::Half> {
  using type = __half;
};

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
  using key_t_ = typename cuda_type<key_t>::type;

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
  using key_t_ = typename cuda_type<key_t>::type;
  using value_t_ = typename cuda_type<value_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);
  const value_t_ *values_in_ = reinterpret_cast<const value_t_*>(values_in);
  value_t_ *values_out_ = reinterpret_cast<value_t_*>(values_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in_, values_out_, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairs,
      keys_in_, keys_out_, values_in_, values_out_, n,
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
  using key_t_ = typename cuda_type<key_t>::type;
  using value_t_ = typename cuda_type<value_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(num_elements * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);
  const value_t_ *values_in_ = reinterpret_cast<const value_t_*>(values_in);
  value_t_ *values_out_ = reinterpret_cast<value_t_*>(values_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceSegmentedRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in_, values_out_,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceSegmentedRadixSort::SortPairs,
      keys_in_, keys_out_, values_in_, values_out_,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

}}}
