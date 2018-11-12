#pragma once

#include "fully_connected_dnnlowp_op.h"

namespace caffe2 {

/**
 * Quantized FC operator with 16-bit accumulation.
 * We'll encounter saturation but this will be faster in Intel CPUs
 */
class FullyConnectedDNNLowPAcc16Op final
  : public FullyConnectedDNNLowPOp<std::uint8_t> {
 public:
  FullyConnectedDNNLowPAcc16Op(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  using BaseType = FullyConnectedDNNLowPOp<std::uint8_t>;

  using BaseType::InputTensorCPU_;
  using BaseType::OutputTensorCPU_;
  using BaseType::dequantize_output_;
  using BaseType::in_qparams_;
  using BaseType::out_qparams_;
  using BaseType::W_quantized_;

 private:
  std::unique_ptr<fbgemm::PackBMatrix<std::int8_t, std::int16_t>>
      Wq_acc16_packed_;

  // Wq outlier in CSC format
  std::unique_ptr<fbgemm::CompressedSparseColumn> Wq_outlier_;
  int nbits_in_non_outlier_;
  int copy_to_32bit_frequency_;
}; // class FullyConnectedDNNLowPAcc16Op

} // namespace caffe2
