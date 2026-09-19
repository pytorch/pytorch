#pragma once

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/cub.cuh>

namespace at::cuda::cub::detail {

template <typename key_t, int value_size>
void radix_sort_pairs_impl(
    const key_t* keys_in,
    key_t* keys_out,
    const OpaqueType<value_size>* values_in,
    OpaqueType<value_size>* values_out,
    int64_t n,
    bool descending,
    int64_t begin_bit,
    int64_t end_bit) {
  // n is passed through as the backend's NumItemsT/Size, which is 64-bit here:
  // CUB selects a 64-bit offset type for it and rocPRIM is templated on it, so
  // there is no INT_MAX ceiling to enforce.
  using key_t_ = typename detail::cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t*>(keys_out_owner.get());
  }

  const key_t_* keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_* keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(
        NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortPairsDescending,
        keys_in_,
        keys_out_,
        values_in,
        values_out,
        n,
        begin_bit,
        end_bit,
        c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(
        NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortPairs,
        keys_in_,
        keys_out_,
        values_in,
        values_out,
        n,
        begin_bit,
        end_bit,
        c10::cuda::getCurrentCUDAStream());
  }
}

#define AT_INSTANTIATE_SORT_PAIRS(key_t, value_size) \
  template void radix_sort_pairs_impl(               \
      const key_t* keys_in,                          \
      key_t* keys_out,                               \
      const OpaqueType<value_size>* values_in,       \
      OpaqueType<value_size>* values_out,            \
      int64_t n,                                     \
      bool descending,                               \
      int64_t begin_bit,                             \
      int64_t end_bit);

#define AT_INSTANTIATE_SORT_PAIRS_8(scalar_t, ScalarType) \
  AT_INSTANTIATE_SORT_PAIRS(scalar_t, 8)

} // namespace at::cuda::cub::detail
