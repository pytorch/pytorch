#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace ops {

int register_vulkan_conv2d_packed_context();
int register_vulkan_conv1d_packed_context();
int register_vulkan_linear_packed_context();
int register_vulkan_layernorm_packed_context();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
