#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

vTensor pack_weights(
    const Tensor& weight_arg) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

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

  for (const auto src_h : c10::irange(src_kh_sz)) {
    for (const auto src_w : c10::irange(src_kw_sz)) {
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
    const Tensor& weight_arg,
    const c10::optional<Tensor>& bias_arg) {
  if (bias_arg && bias_arg->is_vulkan()) {
    return convert(*bias_arg);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

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

    for (const auto src_h : c10::irange(src_kh_sz)) {
      for (const auto src_w : c10::irange(src_kw_sz)) {
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

VulkanOpContext context_create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  TORCH_CHECK(
      available(weight, bias),
      "Vulkan Linear not available! "
      "Reason: The provided (weight, bias) parameters are either invalid "
      "individually or their combination is not supported by Vulkan Impl.");

  c10::impl::GenericList packed_context{c10::AnyType::get()};
  packed_context.reserve(2);
  packed_context.emplace_back(convert(pack_weights(weight)));
  packed_context.emplace_back(convert(pack_biases(weight, bias)));

  c10::impl::GenericList unpacked_context{c10::AnyType::get()};
  unpacked_context.reserve(2);
  unpacked_context.emplace_back(weight);
  unpacked_context.emplace_back(bias);

  return VulkanOpContext::create(packed_context, unpacked_context);
}

Tensor context_run(
    const Tensor& input_arg,
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context,
    const float alpha,
    const float beta,
    const std::string& op_name) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const vTensor& packed_v_weight = convert(packed_context.get(0).toTensor());
  const vTensor& packed_v_bias = convert(packed_context.get(1).toTensor());
  const Tensor& unpacked_weight = unpacked_context.get(0).toTensor();
  const c10::optional<Tensor>& unpacked_bias
   = unpacked_context.get(1).isTensor() ? unpacked_context.get(1).toTensor() : c10::optional<Tensor>();

  TORCH_CHECK(
      usable(input, unpacked_weight, unpacked_bias),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  c10::SmallVector<int64_t, 4u> output_sizes{
      v_input.sizes()[Layout::Parameter::height],
      unpacked_weight.sizes()[Layout::Parameter::width],
  };

  vTensor v_output {
      context,
      {
        v_input.sizes()[Layout::Parameter::height],
        unpacked_weight.sizes()[Layout::Parameter::width],
      },
      input.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if (v_input.has_image() &&
        packed_v_weight.has_image() &&
        packed_v_bias.has_image()) {
      if (unpacked_bias && unpacked_bias->defined()) {
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
              safe_downcast<uint32_t>(div_up(unpacked_weight.sizes()[Layout::Parameter::width], INT64_C(2))),
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
            packed_v_weight.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            packed_v_bias.image(
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
              safe_downcast<uint32_t>(div_up(unpacked_weight.sizes()[Layout::Parameter::width], INT64_C(2))),
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
            packed_v_weight.image(
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

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {

  VulkanOpContext vulkan_context = context_create(weight, bias);

  return context_run(
    input,
    vulkan_context.get_packed(),
    vulkan_context.get_unpacked(),
    alpha.to<float>(),
    beta.to<float>(),
    "aten::addmm");
}

Tensor mm(
    const Tensor& mat1_arg,
    const Tensor& mat2_arg) {

  VulkanOpContext vulkan_context = context_create(
    mat2_arg,
    c10::optional<Tensor>());

  return context_run(
    mat1_arg,
    vulkan_context.get_packed(),
    vulkan_context.get_unpacked(),
    1.0f,
    1.0f,
    "aten::mm");
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));
}

#endif /* USE_VULKAN_API */

} // namespace

VulkanOpContext linear_context_create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  return context_create(weight, bias);
}

Tensor linear_context_run(
    const Tensor& input_arg,
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context,
    const float alpha,
    const float beta,
    const std::string& op_name) {
  return context_run(input_arg, packed_context, unpacked_context, alpha, beta, op_name);
}

c10::intrusive_ptr<VulkanOpContext> create_linear_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias) {
  return c10::make_intrusive<VulkanOpContext>(
      linear_context_create(
          weight,
          bias));
}

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<VulkanOpContext>& vulkan_context) {
  return linear_context_run(
    input,
    vulkan_context->get_packed(),
    vulkan_context->get_unpacked(),
    1.0,
    1.0,
    "prepacked::linear_clamp_run_vulkan");
}

/* Backwards compatibility */
LinearOpContext::LinearOpContext(VulkanOpContext vulkan_context)
  : vulkan_context_{std::move(vulkan_context)} {
}

LinearOpContext LinearOpContext::create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  return LinearOpContext {
      linear_context_create(
        weight,
        bias)
  };
}

Tensor LinearOpContext::run(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const std::string& op_name) const {
  return linear_context_run(
    input_arg,
    vulkan_context_.get_packed(),
    vulkan_context_.get_unpacked(),
    alpha,
    beta,
    op_name);
}

LinearOpContext::State LinearOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ = std::get<1>(vulkan_context_.get_state());
  const Tensor unpacked_weight = unpacked_.get(0).toTensor();
  const c10::optional<Tensor> unpacked_bias
   = unpacked_.get(1).isTensor() ? unpacked_.get(1).toTensor() : c10::optional<Tensor>();
  return LinearOpContext::State{
    unpacked_weight,
    unpacked_bias
  };
}

c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearOpContext>(
      LinearOpContext::create(
          std::move(weight),
          std::move(bias)));
}

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& context) {
  return context->run(input, 1.0, 1.0, "prepacked::linear_clamp_run");
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
