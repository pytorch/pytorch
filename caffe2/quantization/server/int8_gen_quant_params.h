// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"

namespace caffe2 {
using namespace std;
using dnnlowp::TensorQuantizationParams;

struct Int8QuantSchemeBlob {
 public:
  Int8QuantSchemeBlob(std::string quantization_kind, bool preserve_sparsity)
      : quantization_kind_(quantization_kind),
        preserve_sparsity_(preserve_sparsity) {}
  std::string quantization_kind_;
  bool preserve_sparsity_;
};
struct Int8QuantParamsBlob {
 public:
  Int8QuantParamsBlob(float scale, int zero_point) {
    qparam.scale = scale;
    qparam.zero_point = zero_point;
  }
  dnnlowp::TensorQuantizationParams qparam;
};

template <class Context, class Engine = DefaultEngine>
class Int8GenQuantParamsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Int8GenQuantParamsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  bool RunOnDevice() override {
    // Generate Int8 quant params based on the input data (last N samples of the
    // activations) and the quant scheme
    const auto& input_data = Input(0);
    const auto* quant_scheme =
        this->template Input<unique_ptr<Int8QuantSchemeBlob>>(1).get();
    CAFFE_ENFORCE(input_data.dim() > 0);
    CAFFE_ENFORCE(quant_scheme);
    std::string quant_kind = quant_scheme->quantization_kind_;
    bool preserve_sparsity = quant_scheme->preserve_sparsity_;
    dnnlowp::QuantizationFactory* qfactory =
        dnnlowp::QuantizationFactory::GetDefaultInstance();
    TensorQuantizationParams qparam = qfactory->ChooseQuantizationParams(
        input_data.template data<float>(),
        input_data.numel(),
        dnnlowp::StringToKind(quant_kind),
        8,
        preserve_sparsity);
    auto* output_qparam =
        this->template Output<unique_ptr<Int8QuantParamsBlob>>(0);
    output_qparam->reset(
        new Int8QuantParamsBlob(qparam.scale, qparam.zero_point));
    return true;
  }

}; // class Int8GenQuantParamsOp

} // namespace caffe2
