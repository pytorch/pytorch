#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/cub.cuh>

namespace at::cuda::cub {

template <typename key_t>
void radix_sort_keys(
    const key_t* keys_in,
    key_t* keys_out,
    int64_t n,
    bool descending,
    int64_t begin_bit,
    int64_t end_bit) {
  TORCH_CHECK(
      n <= std::numeric_limits<int>::max(),
      "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  const key_t_* keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_* keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(
        NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortKeysDescending,
        keys_in_,
        keys_out_,
        n,
        begin_bit,
        end_bit,
        c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(
        NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortKeys,
        keys_in_,
        keys_out_,
        n,
        begin_bit,
        end_bit,
        c10::cuda::getCurrentCUDAStream());
  }
}

#define AT_INSTATIATE_CUB_TEMPLATES(scalar_t, ScalarType) \
  template void radix_sort_keys(                          \
      const scalar_t* keys_in,                            \
      scalar_t* keys_out,                                 \
      int64_t n,                                          \
      bool descending,                                    \
      int64_t begin_bit,                                  \
      int64_t end_bit);

AT_FORALL_SCALAR_TYPES_AND3(Bool, BFloat16, Half, AT_INSTATIATE_CUB_TEMPLATES)
AT_INSTATIATE_CUB_TEMPLATES(uint16_t, UInt16)
AT_INSTATIATE_CUB_TEMPLATES(uint32_t, UInt32)
AT_INSTATIATE_CUB_TEMPLATES(uint64_t, UInt64)

} // namespace at::cuda::cub
