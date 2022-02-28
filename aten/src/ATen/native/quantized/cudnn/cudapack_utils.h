#include <ATen/native/quantized/packed_params.h>

template <int kSpatialDim = 2>
struct TORCH_API PackedConvWeightCudnn : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightCudnn(
      std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
    //   torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
    //   uint8_t transpose,
    //   std::vector<int32_t> col_offsets,
    //   std::vector<int64_t> kernel,
      std::vector<float> w_scale,
      std::vector<int32_t> w_zp,
      c10::QScheme q_scheme)
      : w(std::move(w)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        // output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        // transpose_(transpose),
        // col_offsets(std::move(col_offsets)),
        // kernel(std::move(kernel)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}

  std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w;
  c10::optional<at::Tensor> bias;
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
  double requantize_multiplier;

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

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
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
