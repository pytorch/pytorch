#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class LinearOpContext final : public torch::jit::CustomClassHolder {
 public:
  static LinearOpContext create(
      api::Resource::Pool& pool,
      const Tensor& weight,
      const c10::optional<Tensor>& bias);

  using State = std::tuple<Tensor, c10::optional<Tensor>>;

  Tensor run(const Tensor& input, float beta, float alpha) const;
  State unpack() const;

 private:
  LinearOpContext(
      api::Resource::Pool& pool,
      const Tensor& weight,
      const c10::optional<Tensor>& bias);

 private:
  struct {
    vTensor v_weight;
    vTensor v_bias;
    bool has_bias;
  } packed_;

  struct {
    Tensor weight;
    c10::optional<Tensor> bias;
    bool has_bias;
  } unpacked_;
};

c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias);

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
