#pragma once

#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"
#include "fbgemm/Fbgemm.h"

namespace caffe2 {

template <typename T>
class FullyConnectedRowWiseDNNLowPOp final
  : public DNNLowPOp<T, FullyConnectedOp<CPUContext>> {
 public:
  FullyConnectedRowWiseDNNLowPOp
    (const OperatorDef& operator_def, Workspace *ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, FullyConnectedOp<CPUContext>);

 private:
  bool GetQuantizationParameters_();

  std::size_t axis_{1};
  std::size_t axis_w_{1};
  vector<std::int64_t> Y_shape_cache_;

  std::vector<dnnlowp::RequantizationParams> rowwise_requantization_params_;
  std::vector<dnnlowp::TensorQuantizationParams> rowwise_qparams_;

  using T_signed = typename std::make_signed<T>::type;

  // used in fast path for T == uint8_t
  std::unique_ptr<fbgemm::PackBMatrix<std::int8_t>> Wq_packed_;
  std::vector<std::uint8_t> X_pack_buf_;

  // used in slow path for T != uint8_t
  std::vector<T_signed> W_quantized_;
  std::vector<std::int32_t> b_quantized_;

  std::vector<std::int32_t> column_offsets_;
  std::vector<std::int32_t> row_offsets_;
  std::vector<std::int32_t> Y_int32_;

  bool is_weight_constant_ = true;
  bool rowwise_weight_quantization_ = true;
}; // class FullyConnectedRowWiseDNNLowPOp

} // namespace caffe2
