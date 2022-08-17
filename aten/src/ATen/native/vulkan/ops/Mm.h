#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class LinearPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;

 public:
  LinearPackedContext(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      const bool fill_unpacked = true);

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct Unpacked final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;

    static constexpr uint32_t NumArgs = 2u;
  };

  /*
   * Assigns a name to each index in the packed list.
   */
  struct Packed final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;
    static constexpr uint32_t WeightSizes = 2u;
    static constexpr uint32_t BiasDefined = 3u;

    static constexpr uint32_t NumArgs = 4u;
  };

  static LinearPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(
        unpacked_.size() == Unpacked::NumArgs,
        "unpacked_ must have ",
        Unpacked::NumArgs,
        " arguments, found ",
        unpacked_.size(),
        "!");

    return unpacked_;
  }
};

/*
 * This function is defined for use in other PackedContexts that store linear op
 * contexts as part of its packed args.
 */
c10::intrusive_ptr<LinearPackedContext> create_packed_linear(
    Tensor weight,
    c10::optional<Tensor> bias,
    const bool fill_unpacked);

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias);

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
