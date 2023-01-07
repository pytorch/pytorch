#pragma once

#include <ATen/Tensor.h>
#include <c10/core/QScheme.h>

#ifdef USE_PYTORCH_QNNPACK
// TODO: Refacto QnnpackUtils.h so as to separate code
// needed for quantized op from the generic qnnpack specific
// quantization utilities.
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <pack_block_sparse.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>

namespace ao {
namespace sparse {

struct TORCH_API PackedLinearWeightQnnp
    : public LinearPackedParamsBase {
  PackedLinearWeightQnnp(const at::Tensor& weight, const c10::optional<at::Tensor>& bias, const int64_t out_features_block_size /* block sparsity size across output_features */, const int64_t in_features_block_size /* block sparsity size across input_features */);
  explicit PackedLinearWeightQnnp(const BCSRSerializationType& serialized);
  c10::optional<at::Tensor> orig_bias_;
  // Separate copy of bias exist so that we can fill in zeros when
  // optional bias does not exist. This is to compy with qnnpack operator that
  // expects bias to be present.
  // In case bias is present bias_ is just a reference to orig_bias_
  at::Tensor bias_;
  c10::QScheme q_scheme_;
  double input_scale_;
  std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix_;
  at::Tensor w_scales_;
  std::vector<uint8_t> w_zero_points_;
  std::vector<float> requantization_scales_;
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      sparse_linear_op_{nullptr};
  int64_t output_channels_;
  int64_t input_channels_;
  // Deserialized Tensors are stored to maintain the lifetime of underlying
  // BCSR data.
  // These are left empty if PackedLinearWeightQnnp is created via prepacking
  // rather than deserializing.
  at::Tensor deserialized_bcsr_row_block_indices_;
  at::Tensor deserialized_bcsr_col_block_indices_;
  at::Tensor deserialized_bcsr_weight_values_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
  }
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
  }

  at::Tensor apply_dynamic(const at::Tensor& input) override;
  at::Tensor apply_dynamic_relu(const at::Tensor& input) override;

  LinearPackedSerializationType unpack() override;

  BCSRSerializationType serialize() override;

  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);

  c10::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(const at::Tensor& input);
};

}}  // namespace ao::sparse

#endif // USE_PYTORCH_QNNPACK
