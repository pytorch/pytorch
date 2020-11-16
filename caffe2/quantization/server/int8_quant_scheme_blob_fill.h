// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/int8_gen_quant_params.h"

namespace caffe2 {
using namespace std;

template <class Context, class Engine = DefaultEngine>
class Int8QuantSchemeBlobFillOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Int8QuantSchemeBlobFillOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  bool RunOnDevice() override {
    std::string quantization_kind =
        this->template GetSingleArgument<std::string>(
            "quantization_kind", "MIN_MAX_QUANTIZATION");
    bool preserve_sparsity =
        this->template GetSingleArgument<bool>("preserve_sparsity", false);

    auto* output_qscheme =
        this->template Output<unique_ptr<Int8QuantSchemeBlob>>(0);
    output_qscheme->reset(
        new Int8QuantSchemeBlob(quantization_kind, preserve_sparsity));
    return true;
  }

}; // class Int8QuantSchemeBlobFillOp

} // namespace caffe2
