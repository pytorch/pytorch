#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const optional<MemoryFormat> memory_format);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
