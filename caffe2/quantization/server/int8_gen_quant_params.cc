// Copyright 2004-present Facebook. All Rights Reserved.
#include "int8_gen_quant_params.h"
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
    .SetDoc(
        R"DOC(Operator wrapper for generating int8 tensor quantization parameters given the input data and quant scheme)DOC");

} // namespace caffe2
