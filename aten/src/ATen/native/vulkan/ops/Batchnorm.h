#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class BatchNormPackedContext final : virtual public VulkanPackedContext,
                                     public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;

 public:
  BatchNormPackedContext(
      const std::optional<Tensor>& weight_opt,
      const std::optional<Tensor>& bias_opt,
      const std::optional<Tensor>& running_mean_opt,
      const std::optional<Tensor>& running_var_opt,
      double eps);

  /*
   * Assigns a name to each index in the packed/unpacked list.
   */
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;
    static constexpr uint32_t kBias = 1u;
    static constexpr uint32_t kRunningMean = 2u;
    static constexpr uint32_t kRunningVar = 3u;
    static constexpr uint32_t kEps = 4u;

    static constexpr uint32_t kNumArgs = 5u;
  };

  static BatchNormPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

    return unpacked_;
  }
};

c10::intrusive_ptr<BatchNormPackedContext> create_batchnorm_context(
    std::optional<Tensor>&& weight_opt,
    std::optional<Tensor>&& bias_opt,
    std::optional<Tensor>&& running_mean_opt,
    std::optional<Tensor>&& running_var_opt,
    bool training,
    double /* momentum */,
    double eps,
    bool /* cudnn_enable, deprecated */);

Tensor run_batchnorm_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<BatchNormPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
