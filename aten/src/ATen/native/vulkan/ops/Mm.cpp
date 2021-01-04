#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

vTensor pack_weights(
  api::Resource::Pool& pool,
  const Tensor& weight_arg) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef w_sizes = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  vTensor v_weight{
      api::context(),
      &pool,
      w_sizes,
      weight.options(),
  };

  {
    using Future = vTensor::Future<void, vTensor::Access::Write>;
    Future v_weight_future = v_weight.host<void, vTensor::Access::Write>();
    Future::Payload v_weight_payload = v_weight_future.wait();

    memcpy(
        v_weight_payload.get(),
        src_weight_ptr,
        std::min(weight.nbytes(), v_weight.nbytes()));
  }

  return v_weight;
}

vTensor pack_biases(
    api::Resource::Pool& pool,
    const Tensor& weight_arg,
    const c10::optional<Tensor>& bias_arg) {
  if (bias_arg && bias_arg->is_vulkan()) {
    return convert(*bias_arg);
  }

  vTensor v_bias{
      api::context(),
      &pool,
      {weight_arg.sizes()[Layout::Parameter::width]},
      weight_arg.options(),
  };

  {
    using Future = vTensor::Future<void, vTensor::Access::Write>;
    Future v_bias_future = v_bias.host<void, vTensor::Access::Write>();
    Future::Payload v_bias_payload = v_bias_future.wait();

    if (bias_arg) {
      memcpy(
          v_bias_payload.get(),
          bias_arg->contiguous().data_ptr<float>(),
          std::min(bias_arg->nbytes(), v_bias.nbytes()));
    } else {
      memset(
          v_bias_payload.get(),
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
  }

  return v_bias;
}

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  return api::available() &&
         // Weight
         (2 == weight.ndimension()) &&
         (weight.size(Layout::Parameter::height) > 0) &&
         (weight.size(Layout::Parameter::width) > 0) &&
         ((c10::DeviceType::CPU == weight.device().type()) ||
          (c10::DeviceType::Vulkan == weight.device().type())) &&
         (kFloat == weight.scalar_type()) &&
         !weight.requires_grad() &&
         // Bias
         ((bias && bias->defined()) ? ((bias->ndimension() > 0) &&
                                       ((c10::DeviceType::CPU == bias->device().type()) ||
                                        (c10::DeviceType::Vulkan == bias->device().type())) &&
                                       (kFloat == bias->scalar_type()) &&
                                       ((bias->ndimension() > 1) ?
                                            (bias->size(Layout::Parameter::width) ==
                                                weight.size(Layout::Parameter::width))
                                            : true) &&
                                       !bias->requires_grad())
                                    : true) &&
         true;
}

bool usable(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& /* bias */) {
  return (2 == input.ndimension()) &&
         (c10::DeviceType::Vulkan == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         (input.size(Layout::Parameter::width) ==
              weight.size(Layout::Parameter::height)) &&
         !input.requires_grad() &&
         true;
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar beta,
    const Scalar alpha) {
  return LinearOpContext::create(
      api::context()->resource().pool,
      weight,
      bias).run(
          input,
          alpha.to<float>(),
          beta.to<float>());
}

Tensor mm(
    const Tensor& mat1_arg,
    const Tensor& mat2_arg) {
  api::Context* const context = api::context();

  const Tensor mat1 = mat1_arg.is_vulkan() ? mat1_arg : mat1_arg.vulkan();
  const vTensor& v_mat1 = convert(mat1);

  const Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  const vTensor& v_mat2 = convert(mat2);

  const auto v_mat1_sizes = v_mat1.sizes();
  const auto v_mat2_sizes = v_mat2.sizes();

  TORCH_CHECK(
      v_mat1_sizes[Layout::Parameter::width] ==
          v_mat2_sizes[Layout::Parameter::height],
      "Incompatible matrix dimensions!");

  vTensor v_output{
      context,
      {
          v_mat1_sizes[Layout::Parameter::height],
          v_mat2_sizes[Layout::Parameter::width],
      },
      mat1.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_mat1.has_image() && v_mat2.has_image()) {
      const struct {
        uvec3 size;
        int32_t K;
      } block {
        v_output.extents(),
        safe_downcast<int32_t>(v_mat1_sizes[Layout::Parameter::width]),
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(mm),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat1.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat2.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("mm", TORCH_FN(mm));
}

#endif /* USE_VULKAN_API */

} // namespace

LinearOpContext::LinearOpContext(
    api::Resource::Pool& pool,
    const Tensor& weight,
    const c10::optional<Tensor>& bias)
  : packed_{
      pack_weights(pool, weight),
      pack_biases(pool, weight, bias),
    },
    unpacked_{
      weight,
      bias,
    } {
}

LinearOpContext LinearOpContext::create(
    api::Resource::Pool& pool,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  TORCH_CHECK(
      available(weight, bias),
      "Vulkan Linear not available! "
      "Reason: The provided (weight, bias) parameters are either invalid "
      "individually or their combination is not supported by Vulkan Impl.");

  // Pass in the originals
  return LinearOpContext{
      pool,
      weight,
      bias,
  };
}

Tensor LinearOpContext::run(
    const Tensor& input_arg,
    const float alpha,
    const float beta) const {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  TORCH_CHECK(
      usable(input, unpacked_.weight, unpacked_.bias),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  vTensor v_output{
      context,
      {
          v_input.sizes()[Layout::Parameter::height],
          packed_.v_weight.sizes()[Layout::Parameter::width],
      },
      input.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_output.has_image() &&
        v_input.has_image() &&
        packed_.v_weight.has_image() &&
        packed_.v_bias.has_image()) {
      const struct {
        uvec3 size;
        int32_t K;
        vec2 multiplier;
      } block {
          v_output.extents(),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Parameter::width]),
          {
            alpha,
            beta,
          },
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(addmm),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          packed_.v_weight.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          packed_.v_bias.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

LinearOpContext::State LinearOpContext::unpack() const {
  return LinearOpContext::State{
      unpacked_.weight,
      unpacked_.bias,
  };
}

c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearOpContext>(
      LinearOpContext::create(
          persistent()->pool,
          std::move(weight),
          std::move(bias)));
}

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& context) {
  return context->run(input, 1.0, 1.0);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
