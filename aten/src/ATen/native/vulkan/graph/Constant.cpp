#include <ATen/native/vulkan/graph/Constant.h>

namespace at {
namespace native {
namespace vulkan {

TensorRef::TensorRef(
    const std::vector<int64_t>& t_sizes,
    api::ScalarType t_dtype,
    const void* const t_data)
    : sizes{}, dtype{t_dtype}, data{t_data} {
  size_t ndim = t_sizes.size();
  sizes.resize(ndim);
  for (int i = 0; i < ndim; ++i) {
    sizes[i] = t_sizes[i];
  }
}

} // namespace vulkan
} // namespace native
} // namespace at
