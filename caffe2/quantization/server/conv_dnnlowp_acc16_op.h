#pragma once

#include "caffe2/quantization/server/conv_dnnlowp_op.h"
#include "fbgemm/Fbgemm.h"

namespace caffe2 {

/**
 * Quantized Conv operator with 16-bit accumulation.
 * We'll encounter saturation but this will be faster in Intel CPUs
 */
template <bool ReluFused = false>
class ConvDNNLowPAcc16Op final
  : public ConvDNNLowPOp<std::uint8_t, ReluFused> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ConvDNNLowPAcc16Op(const OperatorDef& operator_def, Workspace *ws);

  using BaseType = ConvDNNLowPOp<std::uint8_t, ReluFused>;
  using BaseType::InputTensorCPU_;
  using BaseType::OutputTensorCPU_;
  using BaseType::col_buffer_;
  using BaseType::dequantize_output_;
  using BaseType::in_qparams_;
  using BaseType::out_qparams_;
  using BaseType::row_offsets_;
  using BaseType::W_quantized_;
  using BaseType::X_pack_buf_;
  using BaseType::Y_int32_;
  using BaseType::FILTER;
  using BaseType::INPUT;
  using BaseType::BIAS;

 private:
  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  bool GetQuantizationParameters_() override;

  template <typename InType> bool RunOnDeviceWithOrderNCHWAndType_();
  template <typename InType> bool RunOnDeviceWithOrderNHWCAndType_();

  void ConvOutlier_(
      const std::uint8_t* col_buffer,
      vector<std::int32_t>* Y_int32);

  std::vector<std::unique_ptr<fbgemm::PackBMatrix<std::int8_t, std::int16_t>>>
      Wq_acc16_packed_;

  // Wq outlier in CSC format
  std::vector<fbgemm::CompressedSparseColumn> Wq_outlier_;

  int nbits_in_non_outlier_;
  int copy_to_32bit_frequency_;

  bool first_invocation_ = true;
}; // class ConvDNNLowPAcc16Op

} // namespace caffe2
