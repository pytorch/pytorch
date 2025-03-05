#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ACL_ENABLED()

#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/function_info/ActivationLayerInfo.h>
#include <arm_compute/runtime/Allocator.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h>
#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>
#include <arm_compute/runtime/Tensor.h>
#include <array>

namespace at::native::acl_utils {

using ACLDynamicQuantMatmulCacheKey = std::tuple<
    int64_t, // M
    bool, // FUSE_RELU
    int64_t // NUM_THREADS
    >;

enum class ACLDynamicQuantMatmulCacheKeyIndex {
  M,
  FUSE_RELU,
  NUM_THREADS,
};

struct ACLDynamicQuantMatmul {
  arm_compute::Tensor src_s8_tensor;
  arm_compute::Tensor src_fp32_tensor;
  arm_compute::Tensor wei_tensor;
  std::optional<arm_compute::Tensor> bia_tensor;
  arm_compute::Tensor dst_tensor;
  arm_compute::NEQuantizationLayer quant;
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
  arm_compute::NEActivationLayer acl_relu;
  // configuration details for the ACL gemm
  arm_compute::TensorInfo src_s8_tensor_info;
  arm_compute::TensorInfo src_fp32_tensor_info;
  arm_compute::TensorInfo wei_tensor_info;
  std::optional<arm_compute::TensorInfo> bia_tensor_info;
  arm_compute::TensorInfo dst_tensor_info;
  arm_compute::GEMMInfo gemm_info;
  arm_compute::ActivationLayerInfo acl_relu_info{
      arm_compute::ActivationFunction::RELU};

  // key for use in the cache
  ACLDynamicQuantMatmulCacheKey key;

  ~ACLDynamicQuantMatmul() {
    // this will free memory allocated for the quantized src tensor since the
    // allocation happened through ACL: src_s8_tensor.allocator()->allocate()
    src_s8_tensor.allocator()->free();
    // this will not free memory, it will just tell ACL that we're no longer
    // using the pointer
    wei_tensor.allocator()->free();
    if (bia_tensor.has_value()) {
      bia_tensor.value().allocator()->free();
    }
  }
};

} // namespace at::native::acl_utils

struct PackedLinearWeightsACL : public PackedLinearWeightsOnednn {
  using ACLDynamicQuantMatmul = at::native::acl_utils::ACLDynamicQuantMatmul;
  using ACLDynamicQuantMatmulCacheKey =
      at::native::acl_utils::ACLDynamicQuantMatmulCacheKey;
  using ACLDynamicQuantMatmulCacheKeyIndex =
      at::native::acl_utils::ACLDynamicQuantMatmulCacheKeyIndex;
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

  std::shared_ptr<ACLDynamicQuantMatmul> get_acl_dynamic_quant_matmul(
      const ACLDynamicQuantMatmulCacheKey& key) {
    // We're only maintaining a 2 element LRU cache
    // hit first
    if (acl_dynamic_quant_cache[0] != nullptr &&
        acl_dynamic_quant_cache[0]->key == key) {
      return acl_dynamic_quant_cache[0];
    }
    // hit second
    if (acl_dynamic_quant_cache[1] != nullptr &&
        acl_dynamic_quant_cache[1]->key == key) {
      // update LRU
      std::rotate(
          acl_dynamic_quant_cache.begin(),
          acl_dynamic_quant_cache.begin() + 1,
          acl_dynamic_quant_cache.end());
      return acl_dynamic_quant_cache[0];
    }
    // miss -> replace Least Recently Used - i.e. element at index 1
    acl_dynamic_quant_cache[1] = create_acl_dynamic_quant_matmul(key);
    std::rotate(
        acl_dynamic_quant_cache.begin(),
        acl_dynamic_quant_cache.begin() + 1,
        acl_dynamic_quant_cache.end());
    return acl_dynamic_quant_cache[0];
  }

 private:
  // A 2 element (per layer) cache. Given it's not intended to store more than 2
  // elements, we do not need a fancy implementation. The idea behind it is to
  // allow for a (configuration free) fast path for autoregressive
  // transformer-like models which usually involve 2 input tensor shapes; one
  // for the prefill phase and another for the autoregressive phase
  std::array<std::shared_ptr<ACLDynamicQuantMatmul>, 2> acl_dynamic_quant_cache;

  std::shared_ptr<ACLDynamicQuantMatmul> create_acl_dynamic_quant_matmul(
      const ACLDynamicQuantMatmulCacheKey& key) {
    int64_t m =
        std::get<static_cast<int>(ACLDynamicQuantMatmulCacheKeyIndex::M)>(key);
    bool fuse_relu = std::get<static_cast<int>(
        ACLDynamicQuantMatmulCacheKeyIndex::FUSE_RELU)>(key);
    auto acl_gemm = std::make_shared<ACLDynamicQuantMatmul>();
    acl_gemm->key = key;
    acl_gemm->src_fp32_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(k_, m), arm_compute::Format::F32);

    acl_gemm->src_fp32_tensor_info.set_are_values_constant(false);

    acl_gemm->src_s8_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(k_, m),
        1,
        arm_compute::DataType::QASYMM8_SIGNED,
        // TODO: setting the initial offset value to int8_t max instead of zero,
        // because ACL currently skips MatrixBReduction calculation if the
        // source offset at configuration time is zero. This is fixed by this
        // PR: https://review.mlplatform.org/c/ml/ComputeLibrary/+/12820/8 This
        // will be set to the actual src offset value at runtime.
        arm_compute::QuantizationInfo(
            1.0, std::numeric_limits<int8_t>::max(), true));
    acl_gemm->src_s8_tensor_info.set_are_values_constant(false);

    acl_gemm->wei_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(n_, k_),
        1,
        arm_compute::DataType::QASYMM8_SIGNED,
        arm_compute::QuantizationInfo(wei_scale_, wei_zero_point_, true));
    acl_gemm->wei_tensor_info.set_are_values_constant(true);

    // True iff the linear layer has bias && all std::optional bias containers
    // have a value
    bool with_bias{false};
    if (bias_.has_value()) {
      acl_gemm->bia_tensor_info = arm_compute::TensorInfo(
          arm_compute::TensorShape(1, n_), 1, arm_compute::DataType::F32);
      acl_gemm->bia_tensor = arm_compute::Tensor();
      with_bias = true;
    }

    acl_gemm->dst_tensor_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(n_, m), arm_compute::Format::F32);

    // validate that ACL can handle the given problem and inputs.
    if (fuse_relu) {
      arm_compute::Status relu_status =
          arm_compute::NEActivationLayer::validate(
              &acl_gemm->dst_tensor_info,
              &acl_gemm->dst_tensor_info,
              acl_gemm->acl_relu_info);
      if (relu_status.error_code() != arm_compute::ErrorCode::OK) {
        return nullptr;
      }
    }
    arm_compute::Status quant_status =
        arm_compute::NEQuantizationLayer::validate(
            &acl_gemm->src_fp32_tensor_info, &acl_gemm->src_s8_tensor_info);
    if (quant_status.error_code() != arm_compute::ErrorCode::OK) {
      return nullptr;
    }
    arm_compute::Status gemm_status =
        arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            &acl_gemm->src_s8_tensor_info,
            &acl_gemm->wei_tensor_info,
            with_bias ? &acl_gemm->bia_tensor_info.value() : nullptr,
            &acl_gemm->dst_tensor_info,
            acl_gemm->gemm_info);

    if (gemm_status.error_code() != arm_compute::ErrorCode::OK) {
      return nullptr;
    }

    // set the tensor info (i.e. shape, datatype, quant info) for the ACL
    // tensors
    acl_gemm->src_fp32_tensor.allocator()->init(acl_gemm->src_fp32_tensor_info);
    acl_gemm->src_s8_tensor.allocator()->init(acl_gemm->src_s8_tensor_info);
    acl_gemm->wei_tensor.allocator()->init(acl_gemm->wei_tensor_info);
    if (with_bias) {
      acl_gemm->bia_tensor.value().allocator()->init(
          acl_gemm->bia_tensor_info.value());
    }
    acl_gemm->dst_tensor.allocator()->init(acl_gemm->dst_tensor_info);

    // allocate memory only for the quantized tensor, the rest will use memory
    // already avaliable from PyTorch
    acl_gemm->src_s8_tensor.allocator()->allocate();
    // give ACL access to weight and bias pointer
    acl_gemm->wei_tensor.allocator()->import_memory(
        (int8_t*)weight_.get()->get_data_handle());
    if (with_bias) {
      acl_gemm->bia_tensor.value().allocator()->import_memory(
          (float*)bias_.value().get_data_handle());
    }

    // configure
    acl_gemm->quant.configure(
        &acl_gemm->src_fp32_tensor, &acl_gemm->src_s8_tensor);

    acl_gemm->gemm.configure(
        &acl_gemm->src_s8_tensor,
        &acl_gemm->wei_tensor,
        with_bias ? &acl_gemm->bia_tensor.value() : nullptr,
        &acl_gemm->dst_tensor,
        acl_gemm->gemm_info);

    if (fuse_relu) {
      acl_gemm->acl_relu.configure(
          &acl_gemm->dst_tensor,
          &acl_gemm->dst_tensor,
          acl_gemm->acl_relu_info);
    }

    return acl_gemm;
  }
  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range = false);
};

#endif // AT_MKLDNN_ACL_ENABLED()
