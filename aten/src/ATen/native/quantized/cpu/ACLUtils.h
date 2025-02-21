#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ACL_ENABLED()

#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Utils.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>
#include <arm_compute/function_info/ActivationLayerInfo.h>
#include <arm_compute/runtime/Allocator.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h>
#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>
#include <arm_compute/runtime/Tensor.h>
#include <array>

namespace at::native::acl_utils {

using ACLQuantMatmulCacheKey = std::tuple<
    int64_t, // M
    bool, // FUSE_RELU
    int64_t, // NUM_THREADS
    double, // INPUT_SCALE
    int64_t, // INPUT_OFFSET
    double, // OUTPUT_SCALE
    int64_t, // OUTPUT_OFFSET
    bool // SIGNED_INPUT
    >;

enum class ACLQuantMatmulCacheKeyIndex {
  M,
  FUSE_RELU,
  NUM_THREADS,
  INPUT_SCALE,
  INPUT_OFFSET,
  OUTPUT_SCALE,
  OUTPUT_OFFSET,
  SIGNED_INPUT
};

// Abstract interface to share common stuff between static/dynamic ACL matmuls.
struct ACLQuantMatmul {
  arm_compute::Tensor wei_q_tensor;
  std::optional<arm_compute::Tensor> bia_tensor;
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
  arm_compute::TensorInfo wei_q_tensor_info;
  std::optional<arm_compute::TensorInfo> bia_tensor_info;
  arm_compute::GEMMInfo gemm_info;
  arm_compute::ActivationLayerInfo relu_info{
      arm_compute::ActivationFunction::RELU};
  // key for use in the cache
  ACLQuantMatmulCacheKey key;

  ACLQuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      bool has_bias,
      const ACLQuantMatmulCacheKey& cache_key)
      : key(cache_key) {
    wei_q_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_1, weight_dim_0),
        1,
        arm_compute::DataType::QASYMM8_SIGNED,
        arm_compute::QuantizationInfo(weight_scale, -weight_offset, false));

    wei_q_tensor_info.set_are_values_constant(true);

    if (has_bias) {
      bia_tensor_info = arm_compute::TensorInfo(
          arm_compute::TensorShape(1, weight_dim_1),
          1,
          arm_compute::DataType::F32);
      bia_tensor = arm_compute::Tensor();
    }

    wei_q_tensor.allocator()->init(wei_q_tensor_info);
    if (has_bias) {
      bia_tensor.value().allocator()->init(bia_tensor_info.value());
    }
  }

  virtual ~ACLQuantMatmul() {
    // this will not free memory, it will just tell ACL that we're no longer
    // using the pointer
    wei_q_tensor.allocator()->free();
    if (bia_tensor.has_value()) {
      bia_tensor.value().allocator()->free();
    }
  }

  virtual arm_compute::Status validate() = 0;
  virtual void configure() = 0;
};

struct ACLDynamicQuantMatmul : public ACLQuantMatmul {
  arm_compute::Tensor src_q_tensor;
  arm_compute::Tensor src_tensor;
  arm_compute::Tensor dst_tensor;
  arm_compute::NEQuantizationLayer quant;
  // We need a ReLU layer here (unlike static quantization) because the ReLU
  // cannot be "truly" fused with the GEMM through gemm_info in ACL dynamically
  // quantized matmuls.
  arm_compute::NEActivationLayer acl_relu;
  arm_compute::TensorInfo src_q_tensor_info;
  arm_compute::TensorInfo src_tensor_info;
  arm_compute::TensorInfo dst_tensor_info;

  ACLDynamicQuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      bool has_bias,
      const ACLQuantMatmulCacheKey& cache_key)
      : ACLQuantMatmul(
            weight_dim_0,
            weight_dim_1,
            weight_scale,
            weight_offset,
            has_bias,
            cache_key) {
    int64_t m = std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::M)>(key);

    src_q_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_0, m),
        1,
        // ACL dyanamically quantized matmuls only support (signed) int8_t
        arm_compute::DataType::QASYMM8_SIGNED,
        // TODO: setting the initial offset value to int8_t max instead of zero,
        // because ACL currently skips MatrixBReduction calculation if the
        // source offset at configuration time is zero. This is fixed by this
        // PR: https://review.mlplatform.org/c/ml/ComputeLibrary/+/12820/8 This
        // will be set to the actual src offset value at runtime.
        arm_compute::QuantizationInfo(
            1.0, std::numeric_limits<int8_t>::max(), true));
    src_q_tensor_info.set_are_values_constant(false);

    src_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_0, m),
        arm_compute::Format::F32);
    src_tensor_info.set_are_values_constant(false);

    dst_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_1, m), arm_compute::Format::F32);

    src_q_tensor.allocator()->init(src_q_tensor_info);
    src_tensor.allocator()->init(src_tensor_info);
    dst_tensor.allocator()->init(dst_tensor_info);
  }

  ~ACLDynamicQuantMatmul() override {
    // this will free memory allocated for the quantized src tensor since the
    // allocation happened through ACL: src_q_tensor.allocator()->allocate()
    src_q_tensor.allocator()->free();
  }

  arm_compute::Status validate() override {
    const bool fuse_relu =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::FUSE_RELU)>(key);
    if (fuse_relu) {
      arm_compute::Status relu_status =
          arm_compute::NEActivationLayer::validate(
              &dst_tensor_info, &dst_tensor_info, relu_info);
      if (relu_status.error_code() != arm_compute::ErrorCode::OK) {
        return relu_status;
      }
    }
    arm_compute::Status quant_status =
        arm_compute::NEQuantizationLayer::validate(
            &src_tensor_info, &src_q_tensor_info);
    if (quant_status.error_code() != arm_compute::ErrorCode::OK) {
      return quant_status;
    }
    arm_compute::Status gemm_status =
        arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            &src_q_tensor_info,
            &wei_q_tensor_info,
            bia_tensor_info.has_value() ? &bia_tensor_info.value() : nullptr,
            &dst_tensor_info,
            gemm_info);

    return gemm_status;
  }

  void configure() override {
    quant.configure(&src_tensor, &src_q_tensor);

    gemm.configure(
        &src_q_tensor,
        &wei_q_tensor,
        bia_tensor.has_value() ? &bia_tensor.value() : nullptr,
        &dst_tensor,
        gemm_info);

    const bool fuse_relu =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::FUSE_RELU)>(key);
    if (fuse_relu) {
      acl_relu.configure(&dst_tensor, &dst_tensor, relu_info);
    }
  }
};

struct ACLStaticQuantMatmul : public ACLQuantMatmul {
  arm_compute::Tensor src_q_tensor;
  std::optional<arm_compute::Tensor> bia_q_tensor;
  arm_compute::Tensor dst_q_tensor;
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
  arm_compute::NEActivationLayer acl_relu;
  arm_compute::TensorInfo src_q_tensor_info;
  std::optional<arm_compute::TensorInfo> bia_q_tensor_info;
  arm_compute::TensorInfo dst_q_tensor_info;
  arm_compute::GEMMLowpOutputStageInfo output_stage_info;

  ACLStaticQuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      bool has_bias,
      const ACLQuantMatmulCacheKey& cache_key)
      : ACLQuantMatmul(
            weight_dim_0,
            weight_dim_1,
            weight_scale,
            weight_offset,
            has_bias,
            cache_key) {
    const int64_t m =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::M)>(key);
    const int64_t input_zero_point =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::INPUT_OFFSET)>(
            key);
    const double input_scale =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::INPUT_SCALE)>(
            key);
    const int64_t output_zero_point =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::OUTPUT_OFFSET)>(
            key);
    const double output_scale =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::OUTPUT_SCALE)>(
            key);
    const bool fuse_relu =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::FUSE_RELU)>(key);
    const bool signed_input =
        std::get<static_cast<int>(ACLQuantMatmulCacheKeyIndex::SIGNED_INPUT)>(
            key);
    const auto input_acl_datatype = signed_input
        ? arm_compute::DataType::QASYMM8_SIGNED
        : arm_compute::DataType::QASYMM8;

    src_q_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_0, m),
        1,
        input_acl_datatype,
        arm_compute::QuantizationInfo(input_scale, -input_zero_point, false));
    src_q_tensor_info.set_are_values_constant(false);

    if (has_bias) {
      // ACL statically quantized matmul needs the bias in int32_t
      bia_q_tensor_info = arm_compute::TensorInfo(
          arm_compute::TensorShape(1, weight_dim_1),
          1,
          arm_compute::DataType::S32,
          arm_compute::QuantizationInfo(
              1 / (input_scale * weight_scale), 0, false));

      bia_q_tensor = arm_compute::Tensor();
    }

    dst_q_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(weight_dim_1, m),
        1,
        input_acl_datatype,
        arm_compute::QuantizationInfo(output_scale, output_zero_point, false));

    // Setup lowp_gemm output stage
    int output_multiplier;
    int output_shift;
    float multiplier = (input_scale * weight_scale) / output_scale;
    arm_compute::quantization::calculate_quantized_multiplier_less_than_one(
        multiplier, &output_multiplier, &output_shift);

    output_stage_info.type =
        arm_compute::GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    output_stage_info.gemmlowp_multiplier = output_multiplier;
    output_stage_info.gemmlowp_shift = output_shift;
    output_stage_info.gemmlowp_offset = output_zero_point;

    int32_t min_activation = signed_input ? std::numeric_limits<int8_t>::min()
                                          : std::numeric_limits<uint8_t>::min();
    int32_t max_activation = signed_input ? std::numeric_limits<int8_t>::max()
                                          : std::numeric_limits<uint8_t>::max();

    if (fuse_relu) {
      const arm_compute::UniformQuantizationInfo uqinfo =
          dst_q_tensor_info.quantization_info().uniform();
      std::tie(min_activation, max_activation) =
          arm_compute::get_quantized_activation_min_max(
              relu_info, src_q_tensor_info.data_type(), uqinfo);
    }
    output_stage_info.gemmlowp_min_bound = min_activation;
    output_stage_info.gemmlowp_max_bound = max_activation;
    output_stage_info.output_data_type = dst_q_tensor_info.data_type();

    gemm_info.set_gemmlowp_output_stage(output_stage_info);

    if (fuse_relu) {
      gemm_info.set_activation_info(relu_info);
    }

    src_q_tensor.allocator()->init(src_q_tensor_info);
    dst_q_tensor.allocator()->init(dst_q_tensor_info);
    if (has_bias) {
      bia_q_tensor.value().allocator()->init(bia_q_tensor_info.value());
    }
  }

  ~ACLStaticQuantMatmul() override {
    // this will free memory allocated for the quantized bias tensor since the
    // allocation happened through ACL: bia_q_tensor.allocator()->allocate()
    if (bia_q_tensor.has_value()) {
      bia_q_tensor.value().allocator()->free();
    }
  }

  arm_compute::Status validate() override {
    arm_compute::Status gemm_status =
        arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            &src_q_tensor_info,
            &wei_q_tensor_info,
            bia_q_tensor_info.has_value() ? &bia_q_tensor_info.value()
                                          : nullptr,
            &dst_q_tensor_info,
            gemm_info);

    return gemm_status;
  }

  void configure() override {
    gemm.configure(
        &src_q_tensor,
        &wei_q_tensor,
        bia_q_tensor.has_value() ? &bia_q_tensor.value() : nullptr,
        &dst_q_tensor,
        gemm_info);
  }
};

} // namespace at::native::acl_utils

struct PackedLinearWeightsACL : public PackedLinearWeightsOnednn {
  using ACLQuantMatmul = at::native::acl_utils::ACLQuantMatmul;
  using ACLDynamicQuantMatmul = at::native::acl_utils::ACLDynamicQuantMatmul;
  using ACLStaticQuantMatmul = at::native::acl_utils::ACLStaticQuantMatmul;
  using ACLQuantMatmulCacheKey = at::native::acl_utils::ACLQuantMatmulCacheKey;
  using ACLQuantMatmulCacheKeyIndex =
      at::native::acl_utils::ACLQuantMatmulCacheKeyIndex;

  PackedLinearWeightsACL(
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
    wei_zero_point_ = orig_weight_.q_zero_point();
    wei_scale_ = orig_weight_.q_scale();
  }

  int64_t k_;
  int64_t n_;
  int64_t wei_zero_point_;
  double wei_scale_;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmulT> get_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key) {
    return std::dynamic_pointer_cast<ACLQuantMatmulT>(
        fetch_or_create_acl_quant_matmul<ACLQuantMatmulT>(key));
  }

 private:
  // A 2 element (per layer) cache. Given it's not intended to store more than 2
  // elements, we do not need a fancy implementation. The idea behind it is to
  // allow for a (configuration free) fast path for autoregressive
  // transformer-like models which usually involve 2 input tensor shapes; one
  // for the prefill phase and another for the autoregressive phase
  std::array<std::shared_ptr<ACLQuantMatmul>, 2> acl_quant_matmul_cache;

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmul> fetch_or_create_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key) {
    // We're only maintaining a 2 element LRU cache
    // hit first
    if (acl_quant_matmul_cache[0] != nullptr &&
        acl_quant_matmul_cache[0]->key == key) {
      return acl_quant_matmul_cache[0];
    }
    // hit second
    if (acl_quant_matmul_cache[1] != nullptr &&
        acl_quant_matmul_cache[1]->key == key) {
      // update LRU
      std::rotate(
          acl_quant_matmul_cache.begin(),
          acl_quant_matmul_cache.begin() + 1,
          acl_quant_matmul_cache.end());
      return acl_quant_matmul_cache[0];
    }
    // miss -> replace Least Recently Used - i.e. element at index 1
    acl_quant_matmul_cache[1] = create_acl_quant_matmul<ACLQuantMatmulT>(key);
    std::rotate(
        acl_quant_matmul_cache.begin(),
        acl_quant_matmul_cache.begin() + 1,
        acl_quant_matmul_cache.end());
    return acl_quant_matmul_cache[0];
  }

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmulT> create_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range = false);

  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);
};

template <>
inline std::shared_ptr<at::native::acl_utils::ACLDynamicQuantMatmul>
PackedLinearWeightsACL::create_acl_quant_matmul<
    at::native::acl_utils::ACLDynamicQuantMatmul>(
    const at::native::acl_utils::ACLQuantMatmulCacheKey& key) {
  bool with_bias = bias_.has_value();
  auto acl_gemm = std::make_shared<ACLDynamicQuantMatmul>(
      k_, n_, wei_scale_, wei_zero_point_, with_bias, key);

  // validate
  if (acl_gemm->validate().error_code() != arm_compute::ErrorCode::OK) {
    return nullptr;
  }

  // allocate/import memory
  acl_gemm->src_q_tensor.allocator()->allocate();
  acl_gemm->wei_q_tensor.allocator()->import_memory(
      (int8_t*)weight_.get()->get_data_handle());
  if (with_bias) {
    acl_gemm->bia_tensor.value().allocator()->import_memory(
        (float*)bias_.value().get_data_handle());
  }
  // configure
  acl_gemm->configure();
  return acl_gemm;
}

template <>
inline std::shared_ptr<at::native::acl_utils::ACLStaticQuantMatmul>
PackedLinearWeightsACL::create_acl_quant_matmul<
    at::native::acl_utils::ACLStaticQuantMatmul>(
    const at::native::acl_utils::ACLQuantMatmulCacheKey& key) {
  bool with_bias = bias_.has_value();
  auto acl_gemm = std::make_shared<ACLStaticQuantMatmul>(
      k_, n_, wei_scale_, wei_zero_point_, with_bias, key);

  // validate
  if (acl_gemm->validate().error_code() != arm_compute::ErrorCode::OK) {
    return nullptr;
  }

  // allocate/import memory
  acl_gemm->wei_q_tensor.allocator()->import_memory(
      (int8_t*)weight_.get()->get_data_handle());
  if (with_bias) {
    acl_gemm->bia_q_tensor.value().allocator()->allocate();
    float* bias_fp32_buffer = (float*)bias_.value().get_data_handle();
    int32_t* bias_s32_buffer =
        (int32_t*)acl_gemm->bia_q_tensor.value().buffer();
    const float bias_scale =
        acl_gemm->bia_q_tensor_info.value().quantization_info().uniform().scale;
    // Quantize the bias to int32_t. It makes sense to do it here rather in the
    // prepack phase because dynamically quantized ACL matmuls don't need the
    // bias in int32_t.
    at::parallel_for(0, n_, 1, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        bias_s32_buffer[i] =
            int32_t(std::round(bias_fp32_buffer[i] * bias_scale));
      }
    });
  }

  // configure
  acl_gemm->configure();
  return acl_gemm;
}

#endif // AT_MKLDNN_ACL_ENABLED()
