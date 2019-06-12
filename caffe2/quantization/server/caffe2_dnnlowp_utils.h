#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/utils/eigen_utils.h"

namespace dnnlowp {

/**
 * Let consumers of op know that qparams the quantization parameter used
 * for output_index'th output of op.
 */
void PropagateOutputTensorQuantizationParams(
    caffe2::OperatorBase* op, int output_index,
    const TensorQuantizationParams& qparams);

/**
 * If input_index'th input is already quantized, return quantization parameter
 * used for the input tensor (should've been set by
 * PropagateOutputTensorQuantizationParams when the producer was invoked).
 * If the input tensor is not quantized, return the quantization parameter
 * chosen by qfactory based on the distribution of the input tensor
 */
TensorQuantizationParams GetInputTensorQuantizationParamsOf(
    caffe2::OperatorBase* op, int input_index,
    const QuantizationFactory* qfactory, bool is_weight = false);

void SetStaticQuantizationParams(
    caffe2::OperatorBase* op,
    int output_index,
    const TensorQuantizationParams& qparams);

/**
 * @return true if op's outputs should use static quantization (i.e. op has
 *              Y_scale and optionally Y_zero_offset argument).
 */
bool HasStaticQuantization(
    const caffe2::OperatorBase* op, int output_index = 0);

/**
 * Get output_index'th quantization parameter.
 * Should be used only when UseStaticQuantization is true
 */
TensorQuantizationParams GetStaticQuantizationParamsOf(
    const caffe2::OperatorBase* op, int output_index);


/**
 * Quantize input_index'th input if it's not already quantized.
 * a vector temp should be passed to store quantized results.
 *
 * @return array of quantized values
 */
template <typename T>
const T *QuantizeInputIfNeeded(
    caffe2::OperatorBase *op, int input_index,
    const TensorQuantizationParams& qparams, std::vector<T>& temp,
    const QuantizationFactory *qfactory);

template <typename T>
const T *RowWiseQuantizeInputIfNeeded(
    caffe2::OperatorBase *op,
    int input_index,
    const std::vector<TensorQuantizationParams>& qparams,
    std::vector<T>& temp,
    const QuantizationFactory *qfactory);

struct QuantizationErrorStats {
  float sum_sq{0}, sum_err_sq{0};
  float max_abs_err{0};
  // actual and reference values that resulted in max_abs_err
  float max_err_actual{0}, max_err_ref{0};
  int measure_cnt{0};
};

void MeasureQuantizationError(
    const float *actual, const float *ref, size_t len,
    QuantizationErrorStats *stat);

void ReportQuantizationError(
    const caffe2::OperatorBase* op,
    const QuantizationErrorStats& stat);

/**
 * Get QuantizationFactory based on the arguments of op
 */
std::unique_ptr<QuantizationFactory>
    GetQuantizationFactoryOf(const caffe2::OperatorBase* op);

void AdjustOutputTensorQuantizationParamsWithFollowedBy(
    caffe2::OperatorBase* op,
    const std::string& followed_by);

void ParseDNNLowPOperatorArguments(
    caffe2::OperatorBase* op,
    bool* dequantize_output = nullptr,
    bool* measure_quantization_error = nullptr,
    std::string* followed_by = nullptr);

caffe2::NetDef AddScaleZeroOffsetArgumentsWithHistogram(
    caffe2::NetDef net_def, const std::string& histogram_file_name);

} // namespace dnnlowp
