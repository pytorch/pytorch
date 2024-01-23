#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class LayernormPackedContext final : virtual public VulkanPackedContext,
                                     public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;

 public:
  LayernormPackedContext(
      const c10::optional<Tensor>& weight,
      const c10::optional<Tensor>& bias,
      double eps);

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;
    static constexpr uint32_t kBias = 1u;
    static constexpr uint32_t kEps = 2u;

    static constexpr uint32_t kNumArgs = 3u;
  };

  static LayernormPackedContext pack(const c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(!unpacked_.empty(), "unpacked_ does not have any elements!");

    return unpacked_;
  }
};

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    c10::optional<Tensor>&& weight,
    c10::optional<Tensor>&& bias,
    double eps);

Tensor run_layernorm_context(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
