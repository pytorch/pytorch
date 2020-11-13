#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

vTensor pack_weights(api::Resource::Pool& pool, const Tensor& weight_arg) {
  return convert(weight_arg.vulkan());
}

vTensor pack_biases(
    api::Resource::Pool& pool,
    const c10::optional<Tensor>& bias_arg,
    const Tensor& weight_arg) {
  if (bias_arg) {
    return convert(bias_arg->vulkan());
  } else {
    vTensor v_bias{
        api::context(),
        &pool,
        {weight_arg.size(Layout::Parameter::width)},
        weight_arg.options(),
    };

    using Future = vTensor::Future<void, vTensor::Access::Write>;
    Future v_bias_future = v_bias.host<void, vTensor::Access::Write>();
    Future::Payload v_bias_payload = v_bias_future.wait();

    memset(
        v_bias_payload.get(),
        // 2's complement integers and IEEE-754 floating point numbers both
        // have identical bit representations for 0, so can use memset which
        // only accepts uint8_t parameter.
        0,
        v_bias.nbytes());

    return v_bias;
  }
}

bool available(const Tensor& weight, const c10::optional<Tensor>& bias) {
  bool valid = true;
  if (bias && bias->ndimension() > 1) {
    valid =
        (bias->sizes()[Layout::Parameter::width] ==
         weight.sizes()[Layout::Parameter::width]);
  }
  return api::available() && valid;
}

bool usable(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  return (input.sizes()[Layout::Parameter::width] ==
       weight.sizes()[Layout::Parameter::height]);
}

void addmm_impl(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    vTensor& v_output,
    const vTensor& v_self,
    const vTensor& v_mat1,
    const vTensor& v_mat2,
    const float beta,
    const float alpha) {
  if (v_output.has_image() && v_self.has_image() && v_mat1.has_image() &&
      v_mat2.has_image()) {
    const struct {
      float alpha, beta;
    } block{
        alpha,
        beta,
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
        v_output.image(command_buffer, vTensor::Access::Write),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_mat1.image(command_buffer),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_mat2.image(command_buffer),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.image(command_buffer),
        context->resource().pool.uniform(block).object);
  } else {
    TORCH_CHECK(false, "Not implemented!");
  }
}

Tensor addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar beta,
    const Scalar alpha) {
  return LinearOpContext::create(api::context()->resource().pool, mat2, self)
      .run(mat1, beta.to<float>(), alpha.to<float>());
}

Tensor mm(const Tensor& self_arg, const Tensor& mat2_arg) {
  api::Context* const context = api::context();

  const Tensor mat1 = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_mat1 = convert(mat1);

  const Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  const vTensor& v_mat2 = convert(mat2);

  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  TORCH_CHECK(
      mat1_sizes[Layout::Parameter::width] ==
          mat2_sizes[Layout::Parameter::height],
      "Incompatible matrix dimensions!");

  vTensor v_output{
      context,
      {
          mat1_sizes[Layout::Parameter::height],
          mat2_sizes[Layout::Parameter::width],
      },
      mat1.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_mat1.has_image() && v_mat2.has_image()) {
      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          },
          VK_KERNEL(mm),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat1.image(command_buffer),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat2.image(command_buffer));
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
      pack_biases(pool, bias, weight),
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
  TORCH_CHECK(available(weight, bias))
  // Pass in the originals
  return LinearOpContext{
      pool,
      weight,
      bias,
  };
}

Tensor LinearOpContext::run(const Tensor& input_arg, float beta, float alpha)
    const {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  TORCH_CHECK(
      usable(input, unpacked_.weight, unpacked_.bias),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

  vTensor v_output{
      context,
      {
          input_arg.sizes()[Layout::Parameter::height],
          packed_.v_weight.sizes()[Layout::Parameter::width],
      },
      input.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (input_arg.ndimension() == 2) {
      addmm_impl(
          context,
          command_buffer,
          v_output,
          packed_.v_bias,
          v_input,
          packed_.v_weight,
          beta,
          alpha);
    } else {
      TORCH_CHECK(
          false, "linear_run does not yet support inputs with ndim > 2!")
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
  return c10::make_intrusive<LinearOpContext>(LinearOpContext::create(
      persistent()->pool, std::move(weight), std::move(bias)));
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
