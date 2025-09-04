#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Winconsistent-missing-destructor-override")
#include <fbgemm/FbgemmFP16.h>
C10_DIAGNOSTIC_POP()
#include <fbgemm/QuantUtils.h>
C10_DIAGNOSTIC_POP()

// The struct for the packed weight matrix (PackBMatrix) and the corresponding
// column offsets used for the fully connect layer, which are both prepared in
// the prepacking step to save the computations in the inference. Note the
// column offsets include the sum of the B columns as well as the scalar term
// B_zero_point * K, whereas the row offsets created by
// PackAWithQuantRowOffset/PackAWithIm2Col/PackAWithRowOffset are only the sum
// of the A rows. The column offsets are needed for the asymmetric quantization
// (affine quantization) of input matrix.
// Note that in JIT mode we can think of a way to fuse col_offsets with bias.
struct TORCH_API PackedLinearWeight : public LinearPackedParamsBase {
  PackedLinearWeight(
      std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w,
      std::optional<at::Tensor> bias,
      std::vector<int32_t> col_offsets,
      std::vector<float> w_scale,
      std::vector<int32_t> w_zp,
      c10::QScheme q_scheme)
      : w(std::move(w)),
        bias_(std::move(bias)),
        col_offsets(std::move(col_offsets)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(std::move(q_scheme)) {}
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w;
  std::optional<at::Tensor> bias_;
  std::vector<int32_t> col_offsets;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor& apply_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) override;

  at::Tensor& apply_relu_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) override;

  at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) override;

  at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;

  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

 private:
  template <bool ReluFused>
  at::Tensor& apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output);

  template <bool ReluFused>
  at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32_impl(
      const at::Tensor& input,
      double input_scale,
      int64_t input_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range = false);
};

struct TORCH_API PackedLinearWeightFp16 : public LinearPackedParamsBase {
  PackedLinearWeightFp16(
      std::unique_ptr<fbgemm::PackedGemmMatrixFP16> w,
      std::optional<at::Tensor> bias)
      : w(std::move(w)), bias_(std::move(bias)) {}

  std::unique_ptr<fbgemm::PackedGemmMatrixFP16> w;
  std::optional<at::Tensor> bias_;

  at::Tensor apply(
      at::Tensor /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/) override {
    TORCH_INTERNAL_ASSERT(false);
  }
  at::Tensor apply_relu(
      at::Tensor /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/) override {
    TORCH_INTERNAL_ASSERT(false);
  }

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  at::Tensor& apply_dynamic_out(
      const at::Tensor& input,
      at::Tensor& output,
      bool reduce_range = false) override;
  at::Tensor& apply_dynamic_relu_out(
      const at::Tensor& input,
      at::Tensor& output,
      bool reduce_range = false) override;

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

  void set_bias(std::optional<at::Tensor> bias) override;

 private:
  template <bool ReluFused>
  at::Tensor& apply_dynamic_impl(const at::Tensor& input, at::Tensor& output);
};

template <int kSpatialDim = 2>
struct TORCH_API PackedConvWeight : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeight(
      std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      uint8_t transpose,
      std::vector<int32_t> col_offsets,
      std::vector<int64_t> kernel,
      std::vector<float> w_scale,
      std::vector<int32_t> w_zp,
      c10::QScheme q_scheme)
      : w(std::move(w)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        col_offsets(std::move(col_offsets)),
        kernel(std::move(kernel)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}

  std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w;
  std::optional<at::Tensor> bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  uint8_t transpose_;
  std::vector<int32_t> col_offsets;
  std::vector<int64_t> kernel;
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

  at::Tensor apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) override;

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  const float* GetBiasData(at::Tensor* bias);

  void GetQuantizationParams(
      float act_scale,
      float out_scale,
      std::vector<float>* output_multiplier_float,
      std::vector<float>* act_times_w_scale);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return (bool)transpose_;
  }

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

// PackWeight: Convert the weight from uint8 to int8.
inline void convert_uint8_int8(
    int len,
    const uint8_t* src_uint8,
    int8_t* dst_int8) {
  for (const auto i : c10::irange(len)) {
    dst_int8[i] = static_cast<int8_t>(static_cast<int32_t>(src_uint8[i]) - 128);
  }
}

// UnpackWeight: Convert the weight from int8 to uint8.
inline void convert_int8_uint8(
    int len,
    const int8_t* src_int8,
    uint8_t* dst_uint8) {
  for (const auto i : c10::irange(len)) {
    dst_uint8[i] =
        static_cast<uint8_t>(static_cast<int32_t>(src_int8[i]) + 128);
  }
}

namespace at::native::fbgemm_utils {

template <int kSpatialDim = 2>
fbgemm::conv_param_t<kSpatialDim> MakeFbgemmConvParam(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding = std::vector<int>(kSpatialDim, 0),
    bool transposed = false);

// TODO: Remove functions below when ChannelsLast3d is ready.
Tensor MakeStridedQTensorCPU(
    const IntArrayRef& sizes,
    const IntArrayRef& strides,
    const TensorOptions& options,
    QuantizerPtr quantizer);

Tensor MakeEmptyAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    double scale,
    int64_t zero_point);

Tensor MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    const Tensor& scales,
    const Tensor& zero_points);

Tensor ConvertToChannelsLast3dTensor(const Tensor& src);

template <int kSpatialDim = 2>
Tensor TransposeConvTensorUnpackConversion(const Tensor& src, int groups);

template <int kSpatialDim>
Tensor ConvertConvWeightsToChannelLastTensor(
    const at::Tensor& src,
    int groups,
    bool transpose);
} // at::native::namespace fbgemm_utils

#endif // USE_FBGEMM

struct TORCH_API PackedEmbeddingBagWeight : public EmbeddingPackedParamsBase {
  PackedEmbeddingBagWeight(
      at::Tensor packed_w,
      std::vector<float> w_scale,
      std::vector<float> w_zp,
      int64_t bit_rate,
      c10::QScheme q_scheme,
      int64_t version)
      : packed_w(std::move(packed_w)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        bit_rate_(bit_rate),
        q_scheme(q_scheme),
        version_(version) {
    if (!this->packed_w.is_contiguous()) {
      this->packed_w = this->packed_w.contiguous();
    }
  }

  at::Tensor packed_w;
  std::vector<float> w_scale;
  std::vector<float> w_zp;
  int64_t bit_rate_;
  c10::QScheme q_scheme;
  int64_t version_;

  at::Tensor unpack() override;
  static c10::intrusive_ptr<EmbeddingPackedParamsBase> prepack(
      const at::Tensor& weight);

  int64_t bit_rate() const override {
    return bit_rate_;
  }

  int64_t version() const override {
    return version_;
  }

  at::Tensor embeddingbag_byte(
      const at::Tensor& indices,
      const std::optional<at::Tensor>& offsets,
      bool pruned_weights,
      const std::optional<at::Tensor>& per_sample_weights_,
      const std::optional<at::Tensor>& compressed_indices_mapping,
      bool include_last_offset,
      bool is_embedding_op) override;

  at::Tensor embeddingbag_4bit(
      const at::Tensor& indices,
      const std::optional<at::Tensor>& offsets,
      bool pruned_weights,
      const std::optional<at::Tensor>& per_sample_weights_,
      const std::optional<at::Tensor>& compressed_indices_mapping,
      bool include_last_offset,
      bool is_embedding_op) override;
};
