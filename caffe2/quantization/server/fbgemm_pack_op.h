#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/quantization/server/conv_pool_dnnlowp_op_base.h"
#include "caffe2/quantization/server/fbgemm_pack_blob.h"
#include "caffe2/quantization/server/fully_connected_dnnlowp_op.h"

namespace caffe2 {

using FCFp32Op = FullyConnectedOp<CPUContext>;

void QuantizeConvBias(
    const Blob& blob,
    int M,
    const dnnlowp::TensorQuantizationParams& in_qparams,
    const vector<dnnlowp::TensorQuantizationParams>& filter_qparams,
    std::vector<int32_t>& b_quantized, bool use_fp16=false, bool round_nearest_even=true);

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
  bool quantize_channelwise_;
  int nbits_in_non_outlier_; // only for DNNLOWP_ACC16
  bool save_unpacked_weights_;

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
  bool TakeGConvFastPath_();

  fbgemm::conv_param_t<> GetConvParam_();
  fbgemm::conv_param_t<3> GetConv3DParam_();

  // Save quantized weights right after quantization before layout packing for
  // performance purpose
  bool save_unpacked_weights_;
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

int CountOutliers(
    int groups,
    int kernel_dim,
    int M,
    int nbits_in_non_outlier,
    vector<std::int8_t>& W_quantized);

/**
 * @param W_quantized input quantized weight that is not packed yet
 */
fbgemm::CompressedSparseColumn* ExtractOutlierMatrix(
    int groups,
    int kernel_dim,
    int M,
    int nbits_in_non_outlier,
    vector<std::int8_t>& W_quantized);
/*
 * Set up used onnxifi data type constexpr
 * Should always be synced with onnxifi.h
 */
constexpr uint64_t kONNXIFI_DATATYPE_UINT8 = 2;
constexpr uint64_t kONNXIFI_DATATYPE_INT32 = 6;
constexpr uint64_t kONNXIFI_DATATYPE_INT8 = 3;

class Int8ConvDNNLowpPackedWeightBlobShapeFunctions
    : public ExternalTensorFunctionsBase {
 public:
  explicit Int8ConvDNNLowpPackedWeightBlobShapeFunctions()
      : ExternalTensorFunctionsBase() {}
  ~Int8ConvDNNLowpPackedWeightBlobShapeFunctions() override {}
  bool isQuantized() const override {
    return true;
  }
  bool IsSameMetaType(TypeIdentifier id) override;
  void SetupExternalTensorDescriptor(
      const Blob* blob,
      std::vector<std::vector<uint64_t>>* shapes,
      std::vector<std::vector<float>>* all_scales,
      std::vector<std::vector<int32_t>>* all_offsets,
      ExternalTensorDescriptor* desc) override;
  void LoadInfoOfBlob(
      const Blob* blob,
      std::vector<float>* scale,
      std::vector<float>* offset,
      uint32_t* axis) override;
  TypeIdentifier GetTypeMetaId() override;
  TypeMeta GetExternalTensorType(const void* c) override;
  vector<int64_t> GetExternalTensorInfo(
      const void* c,
      size_t* capacity,
      DeviceOption* device) override;
};

class Int8FCDNNLowpPackedWeightBlobShapeFunctions
    : public ExternalTensorFunctionsBase {
 public:
  explicit Int8FCDNNLowpPackedWeightBlobShapeFunctions()
      : ExternalTensorFunctionsBase() {}
  ~Int8FCDNNLowpPackedWeightBlobShapeFunctions() override {}
  bool isQuantized() const override {
    return true;
  }
  bool IsSameMetaType(TypeIdentifier id) override;
  void SetupExternalTensorDescriptor(
      const Blob* blob,
      std::vector<std::vector<uint64_t>>* shapes,
      std::vector<std::vector<float>>* all_scales,
      std::vector<std::vector<int32_t>>* all_offsets,
      ExternalTensorDescriptor* desc) override;
  void LoadInfoOfBlob(
      const Blob* blob,
      std::vector<float>* scale,
      std::vector<float>* offset,
      uint32_t* axis) override;
  TypeIdentifier GetTypeMetaId() override;
  TypeMeta GetExternalTensorType(const void* c) override;
  vector<int64_t> GetExternalTensorInfo(
      const void* c,
      size_t* capacity,
      DeviceOption* device) override;
};

} // namespace caffe2
