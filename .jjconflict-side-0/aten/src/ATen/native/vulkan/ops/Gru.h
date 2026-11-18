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

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct Unpacked final {
    static constexpr uint32_t Params = 0u;
    static constexpr uint32_t hasBiases = 1u;
    static constexpr uint32_t NumLayers = 2u;
    static constexpr uint32_t Dropout = 3u;
    static constexpr uint32_t Train = 4u;
    static constexpr uint32_t Bidirectional = 5u;
    static constexpr uint32_t BatchFirst = 6u;

    static constexpr uint32_t NumArgs = 7u;
  };

  /*
   * Assigns a name to each index in the packed list.
   */
  struct Packed final {
    static constexpr uint32_t LinearContexts = 0u;
    static constexpr uint32_t hasBiases = 1u;
    static constexpr uint32_t NumLayers = 2u;
    static constexpr uint32_t Dropout = 3u;
    static constexpr uint32_t Train = 4u;
    static constexpr uint32_t Bidirectional = 5u;
    static constexpr uint32_t BatchFirst = 6u;

    static constexpr uint32_t NumArgs = 7u;
  };

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
