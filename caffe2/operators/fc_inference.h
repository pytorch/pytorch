#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
std::vector<TensorShape> FCShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in,
    bool pretransposed_weight);

OpSchema::Cost CostInferenceForFC(
    const OperatorDef& def,
    const std::vector<TensorShape>& in,
    bool pretransposed_weight = false);

std::vector<TensorShape> FCGradientShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in,
    bool pretransposed_weight);

OpSchema::Cost CostInferenceForFCGradient(
    const OperatorDef& def,
    const std::vector<TensorShape>& in,
    bool pretransposed_weight);

} // namespace caffe2
