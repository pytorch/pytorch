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

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef w_sizes = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  /* Source */
  const int64_t src_kw_sz = w_sizes[Layout::Parameter::width];
  const int64_t src_kh_sz = w_sizes[Layout::Parameter::height];

  /* Destination */
  const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
  const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
  const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;

  vTensor v_weight{
      context,
      &pool,
      {
        4,
        dst_kh_sz,
        dst_kw_sz,
      },
      weight.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (int64_t src_h = 0; src_h < src_kh_sz; ++src_h) {
    for (int64_t src_w = 0; src_w < src_kw_sz; ++src_w) {
      int64_t dst_plane = 2*(src_h%2) + (src_w%2);
      int64_t dst_index = (src_h/2)*dst_kw_sz + (src_w/2);
      memcpy(
          dst_weight_ptr + dst_plane * dst_plane_sz + dst_index,
          src_weight_ptr + src_h * src_kw_sz + src_w,
          sizeof(float));
    }
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

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  if (bias_arg) {
    const Tensor bias = bias_arg->contiguous();
    const IntArrayRef b_sizes = bias.sizes();
    const float* const src_bias_ptr = bias.data_ptr<float>();

    /* Source */
    int64_t src_kw_sz, src_kh_sz;
    if (bias.sizes().size() == 2) {
      src_kw_sz = b_sizes[Layout::Parameter::width];
      src_kh_sz = b_sizes[Layout::Parameter::height];
    }
    else {
      src_kw_sz = b_sizes[Layout::Parameter::height];
      src_kh_sz = 1;
    }

    /* Destination */
    const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
    const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
    const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;

    vTensor v_bias{
        context,
        &pool,
        {
          4,
          dst_kh_sz,
          dst_kw_sz,
        },
        bias_arg->options(),
    };

    Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
    Future::Payload v_bias_payload = v_bias_future.wait();

    float* const dst_bias_ptr = v_bias_payload.get();
    memset(dst_bias_ptr, 0, v_bias.nbytes());

    for (int64_t src_h = 0; src_h < src_kh_sz; ++src_h) {
      for (int64_t src_w = 0; src_w < src_kw_sz; ++src_w) {
        int64_t dst_plane = 2*(src_h%2) + (src_w%2);
        int64_t dst_index = (src_h/2)*dst_kw_sz + (src_w/2);
        memcpy(
            dst_bias_ptr + dst_plane * dst_plane_sz + dst_index,
            src_bias_ptr + src_h * src_kw_sz + src_w,
            sizeof(float));
      }
    }

    return v_bias;
  }
  else {
    vTensor v_bias{
        api::context(),
        &pool,
        {1},
        weight_arg.options(),
    };
    Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
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

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  return api::available() &&
         // Weight
         (2 == weight.ndimension()) &&
         (weight.size(Layout::Parameter::height) > 0) &&
         (weight.size(Layout::Parameter::width) > 0) &&
         ((weight.device().is_cpu()) ||
          (c10::DeviceType::Vulkan == weight.device().type())) &&
         (kFloat == weight.scalar_type()) &&
         !weight.requires_grad() &&
         // Bias
         ((bias && bias->defined()) ? ((bias->ndimension() > 0) &&
                                       ((bias->device().is_cpu()) ||
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
    const Scalar& beta,
    const Scalar& alpha) {
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
  return LinearOpContext::create(
      api::context()->resource().pool,
      mat2_arg,
      c10::optional<Tensor>()).run(
          mat1_arg,
          1.0f,
          1.0f);
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

  c10::SmallVector<int64_t, 4u> output_sizes{
      v_input.sizes()[Layout::Parameter::height],
      unpacked_.weight.sizes()[Layout::Parameter::width],
  };

  vTensor v_output {
      context,
      {
        v_input.sizes()[Layout::Parameter::height],
        unpacked_.weight.sizes()[Layout::Parameter::width],
      },
      input.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;

  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if (v_input.has_image() &&
        packed_.v_weight.has_image() &&
        packed_.v_bias.has_image()) {
      if (unpacked_.bias && unpacked_.bias->defined()) {
        const struct {
          uvec3 size;
          int32_t K;
          vec2 multiplier;
        } block {
            v_output.extents(),
            safe_downcast<int32_t>(div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
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
            {
              safe_downcast<uint32_t>(div_up(unpacked_.weight.sizes()[Layout::Parameter::width], INT64_C(2))),
              safe_downcast<uint32_t>(div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
              1,
            },
            {8, 8, 1},
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
        const struct {
          uvec3 size;
          int32_t K;
        } block_no_bias {
            v_output.extents(),
            safe_downcast<int32_t>(div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
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
            {
              safe_downcast<uint32_t>(div_up(unpacked_.weight.sizes()[Layout::Parameter::width], INT64_C(2))),
              safe_downcast<uint32_t>(div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
              1,
            },
            {8, 8, 1},
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
            // Object lifetime is managed by the resource pool.
            // It is OK not to keep track of the handle.
            context->resource().pool.uniform(block_no_bias).object);
      }
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

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
