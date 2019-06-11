#pragma once

#include "caffe2/quantization/server/conv_dnnlowp_op.h"
#include "fbgemm/Fbgemm.h"

namespace caffe2 {

/**
 * Quantized Conv operator with 16-bit accumulation.
 * We'll encounter saturation but this will be faster in Intel CPUs
 */
template <bool ReluFused = false>
class ConvDNNLowPAcc16Op final : public ConvDNNLowPOp<std::uint8_t, ReluFused> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ConvDNNLowPAcc16Op(const OperatorDef& operator_def, Workspace* ws);

  using BaseType = ConvDNNLowPOp<std::uint8_t, ReluFused>;
  using BaseType::BIAS;
  using BaseType::col_buffer_;
  using BaseType::FILTER;
  using BaseType::in_qparams_;
  using BaseType::INPUT;
  using BaseType::InputTensorCPU_;
  using BaseType::out_qparams_;
  using BaseType::OutputTensorCPU_;
  using BaseType::row_offsets_;
  using BaseType::W_quantized_;
  using BaseType::X_pack_buf_;
  using BaseType::Y_int32_;

 private:
  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  bool GetQuantizationParameters_();

  template <typename PackAMatrix, fbgemm::QuantizationGranularity Q_GRAN>
  void DispatchFBGEMM_(
      PackAMatrix& packA,
      const std::uint8_t* col_buffer_data,
      vector<std::int32_t>* Y_int32,
      uint8_t* Y_uint8_data);

  void ConvOutlier_(
      const std::uint8_t* col_buffer,
      vector<std::int32_t>* Y_int32);

  virtual bool Acc16() const override {
    return !fallback_to_32_bit_accumulation_;
  }

  std::shared_ptr<fbgemm::PackBMatrix<std::int8_t, std::int16_t>>
      Wq_acc16_packed_;

  // Wq outlier in CSC format
  std::shared_ptr<fbgemm::CompressedSparseColumn> Wq_outlier_;

  // Threshold to decide whether a weight is outlier.
  // For example, if nbits_in_non_outlier_ == 7, w is an outlier if w < -64 or
  // w >= 64.
  // nbits_in_non_outlier_ == 0 means everything is outlier.
  // nbits_in_non_outlier_ == 8 means nothing is outlier.
  int nbits_in_non_outlier_;
  int copy_to_32bit_frequency_;

  bool first_invocation_{true};
  // If outlier matrix is not sparse enough, using 16-bit accumulation won't
  // give speedup due to too much overhead of sparse matrix multiplication or
  // sparse convolution anyway, so fallback to 32-bit accumulation
  bool fallback_to_32_bit_accumulation_{false};
}; // class ConvDNNLowPAcc16Op

} // namespace caffe2
