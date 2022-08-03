#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class GruPackedContext final : virtual public VulkanPackedContext,
                               public torch::jit::CustomClassHolder {
 public:
  GruPackedContext(
      const std::vector<Tensor>& params_cpu, // weights/biases (cpu)
      bool has_biases,
      int64_t num_layers,
      double dropout,
      bool train,
      bool bidirectional,
      bool batch_first);

  static GruPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override;
};

c10::intrusive_ptr<GruPackedContext> create_gru_context(
    std::vector<Tensor>&& params_cpu, // weights/biases (cpu)
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first);

std::tuple<Tensor, Tensor> run_gru_context(
    const Tensor& input_vk,
    const Tensor& hx_vk,
    const c10::intrusive_ptr<GruPackedContext>& vulkan_context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
