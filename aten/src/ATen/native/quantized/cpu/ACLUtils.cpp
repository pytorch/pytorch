#include <ATen/native/quantized/cpu/ACLUtils.h>

#if AT_MKLDNN_ACL_ENABLED()

#include <ATen/Parallel.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Utils.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>

namespace at::native::acl_utils {

QuantMatmul::QuantMatmul(
    int64_t weight_dim_0,
    int64_t weight_dim_1,
    double weight_scale,
    int64_t weight_offset,
    int8_t* weight_ptr,
    std::optional<float*> bias_ptr,
    const QuantMatmulCacheKey& cache_key)
    : key(cache_key) {
  auto wei_q_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_1, weight_dim_0),
      1,
      arm_compute::DataType::QASYMM8_SIGNED,
      arm_compute::QuantizationInfo(weight_scale, -weight_offset, false));
  wei_q_tensor_info.set_are_values_constant(true);
  wei_q_tensor_.allocator()->init(wei_q_tensor_info);
  wei_q_tensor_.allocator()->import_memory(weight_ptr);

  if (bias_ptr.has_value()) {
    auto bia_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(1, weight_dim_1),
        1,
        arm_compute::DataType::F32);
    bia_tensor_ = arm_compute::Tensor();

    bia_tensor_->allocator()->init(bia_tensor_info);
    bia_tensor_->allocator()->import_memory(bias_ptr.value());
  }
  const bool fuse_relu =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::FUSE_RELU)>(key);
  if (fuse_relu) {
    relu_info_ =
        arm_compute::ActivationLayerInfo(arm_compute::ActivationFunction::RELU);
  }
}

QuantMatmul::~QuantMatmul() {
  // this will not free memory, it will just tell ACL that we're no longer
  // using the pointer
  wei_q_tensor_.allocator()->free();
  if (bia_tensor_.has_value()) {
    bia_tensor_->allocator()->free();
  }
}

DynamicQuantMatmul::DynamicQuantMatmul(
    int64_t weight_dim_0,
    int64_t weight_dim_1,
    double weight_scale,
    int64_t weight_offset,
    int8_t* weight_ptr,
    std::optional<float*> bias_ptr,
    const QuantMatmulCacheKey& cache_key)
    : QuantMatmul(
          weight_dim_0,
          weight_dim_1,
          weight_scale,
          weight_offset,
          weight_ptr,
          bias_ptr,
          cache_key) {
  int64_t m = std::get<static_cast<int>(QuantMatmulCacheKeyIndex::M)>(key);

  auto src_q_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_0, m),
      1,
      // ACL dynamically quantized matmuls only support (signed) int8_t
      arm_compute::DataType::QASYMM8_SIGNED,
      // TODO: setting the initial offset value to int8_t max instead of zero,
      // because ACL currently skips MatrixBReduction calculation if the
      // source offset at configuration time is zero. This is fixed by this
      // PR: https://review.mlplatform.org/c/ml/ComputeLibrary/+/12820/8 This
      // will be set to the actual src offset value at runtime.
      arm_compute::QuantizationInfo(
          /*scale=*/1.0,
          /*offset=*/std::numeric_limits<int8_t>::max(),
          /*is_dynamic=*/true));
  src_q_tensor_info.set_are_values_constant(false);

  auto src_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_0, m), arm_compute::Format::F32);
  src_tensor_info.set_are_values_constant(false);

  auto dst_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_1, m), arm_compute::Format::F32);

  src_q_tensor.allocator()->init(src_q_tensor_info);
  src_tensor.allocator()->init(src_tensor_info);
  dst_tensor.allocator()->init(dst_tensor_info);

  src_q_tensor_orig_ =
      at::empty({m, weight_dim_0}, at::device(c10::kCPU).dtype(c10::kQInt8));
  // allocate/import memory
  src_q_tensor.allocator()->import_memory(src_q_tensor_orig_.data_ptr());

  if (relu_info_.has_value()) {
    relu = arm_compute::NEActivationLayer();
  }
}

DynamicQuantMatmul::~DynamicQuantMatmul() {
  // this will not free memory, it will just tell ACL that we're no longer
  // using the pointer
  src_q_tensor.allocator()->free();
}

arm_compute::Status DynamicQuantMatmul::validate() {
  if (relu_info_.has_value()) {
    auto relu_status = arm_compute::NEActivationLayer::validate(
        dst_tensor.info(), dst_tensor.info(), relu_info_.value());
    if (relu_status.error_code() != arm_compute::ErrorCode::OK) {
      return relu_status;
    }
  }
  auto quant_status = arm_compute::NEQuantizationLayer::validate(
      src_tensor.info(), src_q_tensor.info());
  if (quant_status.error_code() != arm_compute::ErrorCode::OK) {
    return quant_status;
  }
  return arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
      src_q_tensor.info(),
      wei_q_tensor_.info(),
      bia_tensor_.has_value() ? bia_tensor_.value().info() : nullptr,
      dst_tensor.info(),
      gemm_info_);
}

void DynamicQuantMatmul::configure() {
  quant.configure(&src_tensor, &src_q_tensor);
  gemm.configure(
      &src_q_tensor,
      &wei_q_tensor_,
      bia_tensor_.has_value() ? &bia_tensor_.value() : nullptr,
      &dst_tensor,
      gemm_info_);
  if (relu.has_value()) {
    relu->configure(&dst_tensor, &dst_tensor, relu_info_.value());
  }
}

StaticQuantMatmul::StaticQuantMatmul(
    int64_t weight_dim_0,
    int64_t weight_dim_1,
    double weight_scale,
    int64_t weight_offset,
    int8_t* weight_ptr,
    std::optional<float*> bias_ptr,
    const QuantMatmulCacheKey& cache_key)
    : QuantMatmul(
          weight_dim_0,
          weight_dim_1,
          weight_scale,
          weight_offset,
          weight_ptr,
          bias_ptr,
          cache_key) {
  const int64_t m =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::M)>(key);
  const int64_t input_zero_point =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::INPUT_OFFSET)>(key);
  const double input_scale =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::INPUT_SCALE)>(key);
  const int64_t output_zero_point =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::OUTPUT_OFFSET)>(key);
  const double output_scale =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::OUTPUT_SCALE)>(key);
  const bool signed_input =
      std::get<static_cast<int>(QuantMatmulCacheKeyIndex::SIGNED_INPUT)>(key);

  const auto input_acl_datatype = signed_input
      ? arm_compute::DataType::QASYMM8_SIGNED
      : arm_compute::DataType::QASYMM8;

  auto src_q_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_0, m),
      1,
      input_acl_datatype,
      arm_compute::QuantizationInfo(input_scale, -input_zero_point, false));
  src_q_tensor_info.set_are_values_constant(false);
  src_q_tensor.allocator()->init(src_q_tensor_info);

  if (bias_ptr.has_value()) {
    auto bia_q_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(1, weight_dim_1),
        1,
        arm_compute::DataType::S32,
        arm_compute::QuantizationInfo(
            1 / (input_scale * weight_scale), 0, false));
    bia_q_tensor_ = arm_compute::Tensor();
    bia_q_tensor_.value().allocator()->init(bia_q_tensor_info);

    float* bias_fp32_buffer = (float*)bia_tensor_.value().buffer();
    bia_q_tensor_orig_ =
        at::empty({m, weight_dim_0}, at::device(c10::kCPU).dtype(c10::kQInt32));
    int32_t* bias_s32_buffer = (int32_t*)bia_q_tensor_orig_.value().data_ptr();
    const float bias_scale =
        bia_q_tensor_info.quantization_info().uniform().scale;
    // Quantize the bias to int32_t. It makes sense to do it here rather in the
    // prepack phase because dynamically quantized ACL matmuls don't need the
    // bias in int32_t.
    at::parallel_for(0, weight_dim_1, 1, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        bias_s32_buffer[i] =
            int32_t(std::round(bias_fp32_buffer[i] * bias_scale));
      }
    });
    bia_q_tensor_.value().allocator()->import_memory(bias_s32_buffer);
  }
  auto dst_q_tensor_info = arm_compute::TensorInfo(
      arm_compute::TensorShape(weight_dim_1, m),
      1,
      input_acl_datatype,
      arm_compute::QuantizationInfo(output_scale, output_zero_point, false));
  dst_q_tensor.allocator()->init(dst_q_tensor_info);

  // Setup lowp_gemm output stage
  int output_multiplier;
  int output_shift;
  float multiplier = (input_scale * weight_scale) / output_scale;
  arm_compute::quantization::calculate_quantized_multiplier_less_than_one(
      multiplier, &output_multiplier, &output_shift);

  arm_compute::GEMMLowpOutputStageInfo output_stage_info;
  output_stage_info.type =
      arm_compute::GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
  output_stage_info.gemmlowp_multiplier = output_multiplier;
  output_stage_info.gemmlowp_shift = output_shift;
  output_stage_info.gemmlowp_offset = output_zero_point;

  int32_t min_activation = signed_input ? std::numeric_limits<int8_t>::min()
                                        : std::numeric_limits<uint8_t>::min();
  int32_t max_activation = signed_input ? std::numeric_limits<int8_t>::max()
                                        : std::numeric_limits<uint8_t>::max();

  if (relu_info_.has_value()) {
    // figure out min, max values for ReLU
    const arm_compute::UniformQuantizationInfo uqinfo =
        dst_q_tensor_info.quantization_info().uniform();
    std::tie(min_activation, max_activation) =
        arm_compute::get_quantized_activation_min_max(
            relu_info_.value(), src_q_tensor_info.data_type(), uqinfo);
    // fuse ReLU with the GEMM
    gemm_info_.set_activation_info(relu_info_.value());
  }
  output_stage_info.gemmlowp_min_bound = min_activation;
  output_stage_info.gemmlowp_max_bound = max_activation;
  output_stage_info.output_data_type = dst_q_tensor_info.data_type();

  gemm_info_.set_gemmlowp_output_stage(output_stage_info);
}

StaticQuantMatmul::~StaticQuantMatmul() {
  // this will not free memory, it will just tell ACL that we're no longer
  // using the pointer
  if (bia_q_tensor_.has_value()) {
    bia_q_tensor_.value().allocator()->free();
  }
}

arm_compute::Status StaticQuantMatmul::validate() {
  return arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
      src_q_tensor.info(),
      wei_q_tensor_.info(),
      bia_q_tensor_.has_value() ? bia_q_tensor_.value().info() : nullptr,
      dst_q_tensor.info(),
      gemm_info_);
}

void StaticQuantMatmul::configure() {
  gemm.configure(
      &src_q_tensor,
      &wei_q_tensor_,
      bia_q_tensor_.has_value() ? &bia_q_tensor_.value() : nullptr,
      &dst_q_tensor,
      gemm_info_);
}

QuantAdd::QuantAdd(
    arm_compute::DataType dtype,
    const std::vector<int64_t>& input_dims,
    double qa_scale,
    int64_t qa_offset,
    double qb_scale,
    int64_t qb_offset,
    double dst_scale,
    int64_t dst_offset) {
  arm_compute::QuantizationInfo qa_qinfo = {
      static_cast<float>(qa_scale), static_cast<int32_t>(qa_offset), false};
  arm_compute::QuantizationInfo qb_qinfo = {
      static_cast<float>(qb_scale), static_cast<int32_t>(qb_offset), false};
  arm_compute::QuantizationInfo qdst_qinfo = {
      static_cast<float>(dst_scale), static_cast<int32_t>(dst_offset), false};

  arm_compute::TensorShape qa_acl_tensor_shape;
  arm_compute::TensorShape qb_acl_tensor_shape;
  arm_compute::TensorShape qdst_acl_tensor_shape;
  for (int i = input_dims.size() - 1; i >= 0; i--) {
    qa_acl_tensor_shape.set(i, input_dims[i], false, true);
    qb_acl_tensor_shape.set(i, input_dims[i], false, true);
    qdst_acl_tensor_shape.set(i, input_dims[i], false, true);
  }
  arm_compute::TensorInfo qa_acl_tensor_info(
      qa_acl_tensor_shape, 1, dtype, qa_qinfo);
  arm_compute::TensorInfo qb_acl_tensor_info(
      qb_acl_tensor_shape, 1, dtype, qb_qinfo);
  arm_compute::TensorInfo qdst_acl_tensor_info(
      qdst_acl_tensor_shape, 1, dtype, qdst_qinfo);

  qa_tensor.allocator()->init(qa_acl_tensor_info);
  qb_tensor.allocator()->init(qb_acl_tensor_info);
  qdst_tensor.allocator()->init(qdst_acl_tensor_info);
}

arm_compute::Status QuantAdd::validate() {
  return q_add.validate(
      qa_tensor.info(), qb_tensor.info(), qdst_tensor.info(), policy);
}

void QuantAdd::configure() {
  q_add.configure(&qa_tensor, &qb_tensor, &qdst_tensor, policy);
}

} // namespace at::native::acl_utils

PackedLinearWeightsACL::PackedLinearWeightsACL(
    std::unique_ptr<ideep::tensor> weight,
    std::optional<ideep::tensor> bias,
    at::Tensor orig_weight,
    std::optional<at::Tensor> orig_bias)
    : PackedLinearWeightsOnednn(
          std::move(weight),
          std::move(bias),
          std::move(orig_weight),
          std::move(orig_bias)) {
  auto w = *(weight_.get());
  k_ = w.get_dim(0);
  n_ = w.get_dim(1);
  weight_zero_point_ = orig_weight_.q_zero_point();
  weight_scale_ = orig_weight_.q_scale();
}

#endif // AT_MKLDNN_ACL_ENABLED()
