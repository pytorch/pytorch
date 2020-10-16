#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor& copy_(Tensor& self, const Tensor& src) {
  // X -> Vulkan
  if (at::kVulkan == self.device().type()) {
    vTensor& v_self = convert(self);

    // CPU -> Vulkan
    if (at::kCPU == src.device().type()) {
      using Future = vTensor::Future<void, vTensor::Access::Write>;
      Future v_self_future = v_self.host<void, vTensor::Access::Write>();
      Future::Payload v_self_payload = v_self_future.wait();

      memcpy(
        v_self_payload.get(),
        src.contiguous().data_ptr<float>(),
        std::min(src.nbytes(), self.nbytes()));
    }
    // Vulkan -> Vulkan
    else if (at::kVulkan == src.device().type()) {
      api::Command::Buffer command_buffer = api::context()->command().pool.allocate();
      command_buffer.begin();

      command_buffer.copy(
          convert(src).buffer(command_buffer),
          v_self.buffer(command_buffer, vTensor::Access::Write));

      command_buffer.end();
      command_buffer.submit(api::context()->gpu().queue);
    }
    else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported!");
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    const vTensor& v_src = convert(src);

    {
      using Future = vTensor::Future<const void, vTensor::Access::Read>;
      const Future v_src_future = v_src.host<const void>();

      // Vulkan -> CPU
      if (at::kCPU == self.device().type()) {
        const Future::Payload v_src_payload = v_src_future.wait();

        memcpy(
          self.data_ptr<float>(),
          v_src_payload.get(),
          std::min(src.nbytes(), self.nbytes()));
      }
      else {
        TORCH_INTERNAL_ASSERT(false, "Unsupported!");
      }
    }

    api::context()->flush();
  }
  else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported!");
  }

  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
