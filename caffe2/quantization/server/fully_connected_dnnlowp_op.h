#pragma once

#include <fbgemm/Fbgemm.h>
#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

template <typename T, bool ReluFused = false>
class FullyConnectedDNNLowPOp
    : public DNNLowPOp<T, FullyConnectedOp<CPUContext>> {
 public:
  FullyConnectedDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, FullyConnectedOp<CPUContext>);

 protected:
  bool GetQuantizationParameters_(float X_scale_=-1.0, int X_zero_point_=0);

  std::size_t axis_{1};
  std::size_t axis_w_{1};
  float X_scale_{-1.0};
  int X_zero_point_{0};
  vector<std::int64_t> Y_shape_cache_;

  std::vector<dnnlowp::RequantizationParams> requantization_params_;
  bool requantization_param_selected_{false};

  // x86 only provides SIMD instructions that multiply a signed integer with an
  // unsigned integer. We use signed for weights.
  using T_signed = typename std::make_signed<T>::type;

  // used in fast path for T == uint8_t
  std::shared_ptr<fbgemm::PackBMatrix<std::int8_t>> Wq_packed_;
  std::vector<std::uint8_t> X_pack_buf_;

  std::vector<std::int32_t> Y_int32_;
  std::vector<dnnlowp::TensorQuantizationParams> filter_qparams_;
  std::vector<float> filter_scales_;
  std::vector<std::int32_t> filter_zero_points_;

  std::vector<float> requantization_multipliers_;
  bool quantize_channelwise_;

  // used in slow path for T != uint8_t
  std::vector<T_signed> W_quantized_;

  // pre-computed biases and offsets
  std::shared_ptr<std::vector<std::int32_t>> b_quantized_;
  const std::int32_t* b_quantized_data_{nullptr};
  std::vector<std::int32_t> row_offsets_;
  std::shared_ptr<std::vector<std::int32_t>> column_offsets_;

  // Dequantized bias populated when input bias is quantized and
  // dequantized_output_ == true
  std::vector<float> b_dequantized_;
  const float* b_dequantized_data_{nullptr};

  bool is_weight_constant_{true};

  float in_qparams0_scale_old_ = 0;
  std::int32_t in_qparams0_zero_point_old_ = 0;
}; // class FullyConnectedDNNLowPOp

} // namespace caffe2
