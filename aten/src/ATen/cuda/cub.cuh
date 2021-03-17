#include <cub/cub.cuh>

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

template<typename key_t, typename value_t>
static inline void sort_pairs(
    key_t *keys_in, key_t *keys_out,
    value_t *values_in, value_t *values_out,
    int64_t n, int64_t start_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  using key_t_ = typename cuda_type<key_t>::type;
  using value_t_ = typename cuda_type<value_t>::type;
  key_t_ *keys_in_ = reinterpret_cast<key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);
  value_t_ *values_in_ = reinterpret_cast<value_t_*>(values_in);
  value_t_ *values_out_ = reinterpret_cast<value_t_*>(values_out);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // Use the sorted order of keys to rearrange the result array
  size_t temp_storage_bytes = 0;
  ::cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes,
    keys_in_, keys_out_, values_in_, values_out_, n,
    start_bit, end_bit, at::cuda::getCurrentCUDAStream());
  auto tmpDataPtr = allocator.allocate(temp_storage_bytes);
  ::cub::DeviceRadixSort::SortPairs(
    tmpDataPtr.get(), temp_storage_bytes,
    keys_in_, keys_out_, values_in_, values_out_, n,
    start_bit, end_bit, at::cuda::getCurrentCUDAStream());
}

}}}