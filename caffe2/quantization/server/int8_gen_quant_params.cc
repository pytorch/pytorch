// Copyright 2004-present Facebook. All Rights Reserved.
#include "caffe2/quantization/server/int8_gen_quant_params.h"
#include <functional>

namespace caffe2 {
using namespace std;
using namespace dnnlowp;
// Expilictly register TypeMeta
CAFFE_KNOWN_TYPE(unique_ptr<Int8QuantSchemeBlob>);
CAFFE_KNOWN_TYPE(unique_ptr<Int8QuantParamsBlob>);

REGISTER_CPU_OPERATOR(
    Int8GenQuantParams,
    Int8GenQuantParamsOp<CPUContext, DefaultEngine>);
OPERATOR_SCHEMA(Int8GenQuantParams)
    .NumInputs(2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      out[0].add_dims(1);
      return out;
    })
    .Input(
        0,
        "X",
        "The input data, or last N samples of the output activations.")
    .Input(
        1,
        "quant_scheme",
        "Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params.")
    .Output(
        0,
        "quant_param",
        "Int8QuantParamsBlob that contains the scale and zero_point info in TensorQuantizationParams type.")
    .SetDoc(
        R"DOC(Operator wrapper for generating int8 tensor quantization parameters given the input data and quant scheme)DOC");

} // namespace caffe2
