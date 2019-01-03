#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/quantization/server/conv_pool_dnnlowp_op_base.h"
#include "caffe2/quantization/server/fbgemm_pack_blob.h"
#include "caffe2/quantization/server/fully_connected_dnnlowp_op.h"

namespace caffe2 {

using FCFp32Op = FullyConnectedOp<CPUContext>;

class FullyConnectedDNNLowPPackWeightOp final
    : public DNNLowPOp<std::uint8_t, FCFp32Op> {
 public:
  FullyConnectedDNNLowPPackWeightOp(
      const OperatorDef& operator_def,
      Workspace* ws);
  USE_OPERATOR_FUNCTIONS(CPUContext);

  bool RunOnDevice() override;

 private:
  int axis_w_;
  int nbits_in_non_outlier_; // only for DNNLOWP_ACC16

  INPUT_TAGS(FILTER, BIAS);
};

using ConvFp32Op = ConvOp<float, CPUContext>;

/**
 * Pack a weight matrix that can be used by DNNLOWP Int8Conv operators.
 * DNNLOWP operators can pack matrix on demand during their first invocations
 * but calling this operator to pre-pack can have benefits like saving memory
 * space when multiple operators are sharing the same weight.
 * This operator should be a part of init net to be called once to populate
 * packed blob to be used by Int8Conv DNNLOWP operators in the predictor net
 *
 * This operator optionally can also pre-quantize bias.
 * Then, we should also provide the scale of input activation tensor as in_scale
 * argument.
 */
class ConvDNNLowPPackWeightOp final
    : public ConvPoolDNNLowPOpBase<std::uint8_t, ConvFp32Op> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  USE_CONV_POOL_DNNLOWP_OPERATOR_BASE_FUNCTIONS(std::uint8_t, ConvFp32Op);
  ConvDNNLowPPackWeightOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  bool TakeDepthWise3x3FastPath_();
  bool TakeDepthWise3x3x3FastPath_();

  bool quantize_groupwise_;
  int nbits_in_non_outlier_; // only for DNNLOWP_ACC16

  INPUT_TAGS(FILTER, BIAS);
};

// Helper functions for packing weights that can be used by
// ConvDNNLowPAcc16PackWeightOp, ConvDNNLowPOp, and ConvDNNLowPAcc16Op

template <typename T>
void QuantizeWeight(
    const Blob& blob,
    int kernel_dim,
    int M,
    vector<dnnlowp::TensorQuantizationParams>& qparams,
    vector<typename std::make_signed<T>::type>& w_quantized,
    dnnlowp::QuantizationFactory* qfactory);

template <typename T>
void ComputeColumnOffsets(
    int num_rows,
    int num_cols,
    const T* W,
    const vector<dnnlowp::TensorQuantizationParams>& qparams,
    vector<int32_t>& col_offsets);

/**
 * @param W_quantized input quantized weight that is not packed yet
 */
fbgemm::CompressedSparseColumn* ExtractOutlierMatrix(
    int groups,
    int kernel_dim,
    int M,
    int nbits_in_non_outlier,
    vector<std::int8_t>& W_quantized);

} // namespace caffe2
