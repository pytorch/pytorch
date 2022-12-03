#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/Context.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
using namespace at::native::vulkan::ops;

vTensor pack_weights(const Tensor& weight_arg) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();

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

  api::StorageBuffer staging(context, at::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    float* dst_weight_ptr = mapping.template data<float>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_h : c10::irange(src_kh_sz)) {
      for (const auto src_w : c10::irange(src_kw_sz)) {
        int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
        int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
        memcpy(
            dst_weight_ptr + dst_plane * dst_plane_sz + dst_index,
            src_weight_ptr + src_h * src_kw_sz + src_w,
            sizeof(float));
      }
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_biases(
    const Tensor& weight_arg,
    const c10::optional<Tensor>& bias_arg) {
  if (bias_arg && bias_arg->is_vulkan()) {
    return convert(*bias_arg);
  }

  api::Context* const context = api::context();

  if (bias_arg) {
    const Tensor bias = bias_arg->contiguous();
    const IntArrayRef b_sizes = bias.sizes();
    const float* const src_bias_ptr = bias.data_ptr<float>();

    /* Source */
    int64_t src_kw_sz, src_kh_sz;
    if (bias.sizes().size() == 2) {
      src_kw_sz = b_sizes[Layout::Parameter::width];
      src_kh_sz = b_sizes[Layout::Parameter::height];
    } else {
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

    api::StorageBuffer staging(context, at::kFloat, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* dst_bias_ptr = mapping.template data<float>();

      memset(dst_bias_ptr, 0, v_bias.nbytes());

      for (const auto src_h : c10::irange(src_kh_sz)) {
        for (const auto src_w : c10::irange(src_kw_sz)) {
          int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
          int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
          memcpy(
              dst_bias_ptr + dst_plane * dst_plane_sz + dst_index,
              src_bias_ptr + src_h * src_kw_sz + src_w,
              sizeof(float));
        }
      }
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  } else {
    vTensor v_bias{
        api::context(),
        {1},
        weight_arg.options(),
    };

    api::StorageBuffer staging(context, at::kFloat, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* data_ptr = mapping.template data<float>();

      memset(
          data_ptr,
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  }
}

bool available(const Tensor& weight, const c10::optional<Tensor>& bias) {
  return api::available() &&
      // Weight
      (2 == weight.ndimension()) &&
      (weight.size(Layout::Parameter::height) > 0) &&
      (weight.size(Layout::Parameter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type()) && !weight.requires_grad() &&
      // Bias
      ((bias && bias->defined())
           ? ((bias->ndimension() > 0) &&
              ((bias->device().is_cpu()) ||
               (c10::DeviceType::Vulkan == bias->device().type())) &&
              (kFloat == bias->scalar_type()) &&
              ((bias->ndimension() > 1)
                   ? (bias->size(Layout::Parameter::width) ==
                      weight.size(Layout::Parameter::width))
                   : true) &&
              !bias->requires_grad())
           : true) &&
      true;
}

bool usable(const Tensor& input, const IntArrayRef unpacked_weight_sizes) {
  return (2 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::Parameter::width) ==
       unpacked_weight_sizes[Layout::Parameter::height]) &&
      !input.requires_grad() && true;
}

static Tensor reshape_to_2d(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 2,
      "Vulkan Linear op only supports input tensor with dim >= 2");
  const IntArrayRef input_sizes = input_arg.sizes();
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  return input_arg.reshape({d, input_arg.size(-1)});
}

Tensor run_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const vTensor& v_input = convert(input);

  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();
  const bool bias_defined =
      linear_context->get_val(LinearPackedContext::Packed::BiasDefined)
          .toBool();

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  vTensor v_output{
      context,
      {
          v_input.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      input.options(),
  };

  if (bias_defined) {
    const struct {
      uvec3 size;
      int32_t K;
      vec2 multiplier;
    } block{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        {
            alpha,
            beta,
        },
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(addmm),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                unpacked_weight_sizes[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(
                div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  } else {
    const struct {
      uvec3 size;
      int32_t K;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
    };

    api::UniformParamsBuffer params(context, block_no_bias);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(mm),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                unpacked_weight_sizes[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(
                div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }

  Tensor output = convert(v_output);
  if (input_arg.dim() == 2) {
    return output;
  } else {
    std::vector<int64_t> shape;
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    return output.reshape(shape);
  }
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  return run_addmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias)));
}

Tensor mm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  return run_addmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(mat2_arg, c10::optional<Tensor>())));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));
}

#endif /* USE_VULKAN_API */

} // namespace

LinearPackedContext::LinearPackedContext(
    const Tensor& weight,
    const c10::optional<Tensor>& bias)
    : unpacked_{c10::AnyType::get()} {
  TORCH_CHECK(
      available(weight, bias),
      "Vulkan Linear not available! "
      "Reason: The provided (weight, bias) parameters are either invalid "
      "individually or their combination is not supported by Vulkan Impl.");

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(convert(pack_weights(weight)));
  packed_.emplace_back(convert(pack_biases(weight, bias)));
  packed_.emplace_back(weight.sizes());
  packed_.emplace_back(bias && bias->defined());

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
  }
}

LinearPackedContext LinearPackedContext::pack(c10::impl::GenericList unpacked) {
  return LinearPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(weight, bias));
}

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(input, 1.0f, 1.0f, linear_context);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
