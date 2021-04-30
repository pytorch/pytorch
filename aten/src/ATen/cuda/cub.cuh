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
#define CUB_WRAPPER(func, ...) do {                                        \
  size_t temp_storage_bytes = 0;                                           \
  func(nullptr, temp_storage_bytes, __VA_ARGS__);                          \
  auto temp_storage = allocator->allocate(temp_storage_bytes);             \
  func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);               \
  AT_CUDA_CHECK(cudaGetLastError());                                       \
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

template<typename key_t>
static inline void sort_keys(
    const key_t *keys_in, key_t *keys_out,
    int64_t n, bool descending=false, int64_t start_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  using key_t_ = typename cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortKeysDescending,
      keys_in_, keys_out_, n,
      start_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortKeys,
      keys_in_, keys_out_, n,
      start_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template<typename key_t, typename value_t>
static inline void sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t n, bool descending=false, int64_t start_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  using key_t_ = typename cuda_type<key_t>::type;
  using value_t_ = typename cuda_type<value_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;
  c10::DataPtr values_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }
  if (values_out == nullptr) {
    values_out_owner = allocator->allocate(n * sizeof(value_t));
    values_out = reinterpret_cast<value_t *>(values_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);
  const value_t_ *values_in_ = reinterpret_cast<const value_t_*>(values_in);
  value_t_ *values_out_ = reinterpret_cast<value_t_*>(values_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in_, values_out_, n,
      start_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceRadixSort::SortPairs,
      keys_in_, keys_out_, values_in_, values_out_, n,
      start_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
void inclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
  for (int64_t i = 0; i < num_items; i += max_cub_size) {
    int size_cub = std::min<int64_t>(num_items - i, max_cub_size);
    Tensor first_elem; // need to save it for all iterations other than first
    if (i > 0) {
      // need to temporarily transform first element of the range we are
      // operating on; self might be multi-d, but we need to index a single
      // element
      auto self_view = at::_unsafe_view(self, -1);
      first_elem = self_view[i].clone();
      transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
          self.data_ptr<scalar_t>() + i,
          result.data_ptr<scalar_t>() + i - 1,
          self.data_ptr<scalar_t>() + i,
          scan_op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::InclusiveScan(
        input + i,
        output + i,
        scan_op,
        size_cub,
        at::cuda::getCurrentCUDAStream()));
    if (i > 0) {
      if (self.data_ptr<scalar_t>() != result.data_ptr<scalar_t>()) {
        // restore modified first element only if it's not an inplace operation
        auto self_view = at::_unsafe_view(self, -1);
        self_view[i].copy_(first_elem, /*non_blocking=*/true);
      }
    }
  }
}

template<typename InputIteratorT, typename OutputIteratorT>
void exclusive_sum(InputIteratorT input, OutputIteratorT output, int64_t n) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  CUB_WRAPPER(NO_ROCM(detail)::cub::DeviceScan::ExclusiveSum,
    input, output, n, c10::cuda::getCurrentCUDAStream());
}

}}}
