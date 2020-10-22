#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor& binary_(
    const api::Shader::Descriptor& shader_descriptor,
    vTensor& self,
    const vTensor& other,
    const Scalar alpha) {
  api::Context* const context = api::context();

  TORCH_INTERNAL_ASSERT(
      self.is_vulkan(),
      "Vulkan: In-place binary operations on non-Vulkan tensors are not supported!");

  vTensor& v_self = convert(self);

  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image() && v_other.has_image()) {
      const struct {
      } block {
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          {
            8, 8, 1,
          },
          v_self.extents(),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(command_buffer, vTensor::Access::Read | vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_other.image(command_buffer),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      // TODO: Convert all to buffer and dispatch a buffer-only kernel.
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

}

Tensor binary(
    const api::Shader::Descriptor& shader_descriptor,
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar alpha) {
  api::Context* const context = api::context();

  // const Tensor self = self_.is_vulkan() ? self_ : self_.vulkan();
  // const vTensor& v_self = convert(self);

  // const Tensor other = other_.is_vulkan() ? other_ : other_.vulkan();
  // const vTensor& v_other = convert(other);

  // api::Command::Buffer command_buffer = context->command().pool.allocate();
  // command_buffer.begin();
  // {
  //   if (v_output.has_image() && v_self.has_image() && v_other.has_image()) {
  //     const struct {
  //     } block {
  //     };

  //     context->dispatch(
  //         command_buffer,
  //         {
  //           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
  //           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //           VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
  //         },
  //         shader_descriptor,
  //         {
  //           8, 8, 1,
  //         },
  //         v_output.extents(),
  //         // Write-only access bypasses synchronization but inserts appropriate
  //         // barriers if necessary.
  //         v_output.image(command_buffer, vTensor::Access::Write),
  //         // Read-only access is implied on const tensors and triggers an async
  //         // synchronization if necessary.
  //         v_self.image(command_buffer),
  //         // Read-only access is implied on const tensors and triggers an async
  //         // synchronization if necessary.
  //         v_other.image(command_buffer),
  //         // Object lifetime is managed by the resource pool.
  //         // It is OK not to keep track of the handle.
  //         context->resource().pool.uniform(block).object);
  //   }
  //   else {
  //     // TODO: Convert all to buffer and dispatch a buffer-only kernel.
  //   }
  // }
  // command_buffer.end();
  // command_buffer.submit(context->gpu().queue);

  // return convert(v_output);
}

// Tensor& binary_(
//     const api::Shader::Descriptor& shader_descriptor,
//     Tensor& self,
//     const Tensor& other_,
//     const Scalar alpha) {
//   api::Context* const context = api::context();

//   TORCH_INTERNAL_ASSERT(
//       self.is_vulkan(),
//       "Vulkan: In-place binary operations on non-Vulkan tensors are not supported!");

//   vTensor& v_self = convert(self);

//   const Tensor other = other_.is_vulkan() ? other_ : other_.vulkan();
//   const vTensor& v_other = convert(other);

//   api::Command::Buffer command_buffer = context->command().pool.allocate();
//   command_buffer.begin();
//   {
//     if (v_self.has_image() && v_other.has_image()) {
//       const struct {
//       } block {
//       };

//       context->dispatch(
//           command_buffer,
//           {
//             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
//             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
//             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
//           },
//           shader_descriptor,
//           {
//             8, 8, 1,
//           },
//           v_self.extents(),
//           // Read-Write access triggers an async synchronization if necessory
//           // and inserts appropriate barriers if hazards are detected.
//           v_self.image(command_buffer, vTensor::Access::Read | vTensor::Access::Write),
//           // Read-only access is implied on const tensors and triggers an async
//           // synchronization if necessary.
//           v_other.image(command_buffer),
//           // Object lifetime is managed by the resource pool.
//           // It is OK not to keep track of the handle.
//           context->resource().pool.uniform(block).object);
//     }
//     else {
//       // TODO: Convert all to buffer and dispatch a buffer-only kernel.
//     }
//   }
//   command_buffer.end();
//   command_buffer.submit(context->gpu().queue);

//   return self;
// }

Tensor add_scalar(
    const Tensor& self,
    const Scalar other,
    const Scalar alpha) {
  return convert(vTensor{api::context(), {}, {}});
}

Tensor& add_scalar_(
    Tensor& self,
    const Scalar other,
    const Scalar alpha) {
  return self;
}

Tensor add_tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar alpha) {
  return convert(vTensor{api::context(), {}, {}});
}

Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other,
    const Scalar alpha) {
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("add.Scalar", TORCH_FN(add_scalar));
  m.impl("add_.Scalar", TORCH_FN(add_scalar_));
  m.impl("add.Tensor", TORCH_FN(add_tensor));
  m.impl("add_.Tensor", TORCH_FN(add_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
