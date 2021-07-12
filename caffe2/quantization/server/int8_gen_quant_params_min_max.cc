// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/quantization/server/int8_gen_quant_params_min_max.h"
#include <functional>
#include "caffe2/quantization/server/int8_gen_quant_params.h"

namespace caffe2 {
using namespace std;
using namespace dnnlowp;

REGISTER_CPU_OPERATOR(
    Int8GenQuantParamsMinMax,
    Int8GenQuantParamsMinMaxOp<CPUContext, DefaultEngine>);
OPERATOR_SCHEMA(Int8GenQuantParamsMinMax)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& /* in */) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      out[0].add_dims(1);
      return out;
    })
    .Input(0, "min", "The lower bound of the tensor to be quantized.")
    .Input(1, "max", "The upper bound of the tensor to be quantized.")
    .Input(
        2,
        "quant_scheme",
        "(Optional) Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params. We only use preserve_sparsity in this op which is default to be false.")
    .Output(
        0,
        "quant_param",
        "Int8QuantParamsBlob that contains the scale and zero_point info in TensorQuantizationParams type.")
    .SetDoc(
        R"DOC(Operator wrapper for generating int8 tensor quantization parameters given lower and upper bound of the input tensor)DOC");

} // namespace caffe2
