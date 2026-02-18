#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/SortImpl.cuh>
namespace at::native {
  INSTANTIATE_SORT_COMMON(uint8_t)
}
