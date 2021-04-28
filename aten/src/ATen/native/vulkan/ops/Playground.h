#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class PlaygroundOpContext final : public torch::jit::CustomClassHolder {
 public:
  static PlaygroundOpContext create(const Tensor& test);

  using State = std::tuple<Tensor>;

  Tensor run(const Tensor& input);
  State unpack() const;

  void fill_image(const api::Resource::Buffer& buffer, api::Resource::Fence& fence);

 private:
  PlaygroundOpContext(const Tensor& test);

 private:
  struct {
    vTensor v_test;
  } packed_;

  struct {
    Tensor test;
  } unpacked_;

  VkDescriptorSet descriptor_set;
  api::Command::Buffer cmd_buffer;
  api::Resource::Buffer in_buffer;
  bool initted = false;
};

c10::intrusive_ptr<PlaygroundOpContext> playground_prepack(Tensor&& test);

Tensor playground_run(
    const Tensor& input,
    const c10::intrusive_ptr<PlaygroundOpContext>& context);


} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
