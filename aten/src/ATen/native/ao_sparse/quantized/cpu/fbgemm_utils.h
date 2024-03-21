#pragma once

#include <ATen/Tensor.h>
#include <c10/core/QScheme.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmSparse.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>

namespace ao {
namespace sparse {

struct TORCH_API PackedLinearWeight
    : public LinearPackedParamsBase {
  PackedLinearWeight(std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w,
                     c10::optional<at::Tensor> bias,
                     std::vector<int32_t> col_offsets,
                     std::vector<float> w_scale,
                     std::vector<int32_t> w_zp,
                     c10::QScheme q_scheme,
                     const int64_t out_features_block_size /* block sparsity size across output_features */,
                     const int64_t in_features_block_size /* block sparsity size across input_features */)
      : LinearPackedParamsBase(
            out_features_block_size,
            in_features_block_size),
        w(std::move(w)),
        bias_(std::move(bias)),
        col_offsets(std::move(col_offsets)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}
  std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w;
  c10::optional<at::Tensor> bias_;
  std::vector<int32_t> col_offsets;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }
  at::Tensor apply_dynamic_relu(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }

  LinearPackedSerializationType unpack() override;

  BCSRSerializationType serialize() override;

  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);

  c10::optional<at::Tensor> bias() override {
    return bias_;
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
};

}}  // namespace ao::sparse

#endif // USE_FBGEMM

namespace ao {
namespace sparse {
int register_linear_params();
}}  // namespace ao::sparse
