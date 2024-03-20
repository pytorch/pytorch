// Copyright 2004-present Facebook. All Rights Reserved.
#include "caffe2/quantization/server/int8_quant_scheme_blob_fill.h"
#include <functional>

namespace caffe2 {
using namespace std;
using namespace dnnlowp;

REGISTER_CPU_OPERATOR(
    Int8QuantSchemeBlobFill,
    Int8QuantSchemeBlobFillOp<CPUContext, DefaultEngine>);
OPERATOR_SCHEMA(Int8QuantSchemeBlobFill)
    .NumInputs(0)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(TensorProto_DataType_STRING);
      out[0].add_dims(1);
      return out;
    })
    .Arg(
        "quantization_kind",
        "The kind of quant scheme that would be used to generate quant param")
    .Arg("preserve_sparsity", "Flag to preserve sparsity or not")
    .Output(
        0,
        "quant_scheme",
        "Int8QuantSchemeBlob that specifies the quantization kind and preserve_sparsity options when generating the quant params.")
    .SetDoc(
        R"DOC(Operator wrapper for generating int8 quant scheme blob given the preserve sparsity and quantization kind)DOC");

} // namespace caffe2
