// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/int8_gen_quant_params.h"
#include <glog/logging.h>


namespace caffe2 {
using namespace std;
using dnnlowp::TensorQuantizationParams;

template <class Context, class Engine = DefaultEngine>
class Int8GenQuantParamsMinMaxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Int8GenQuantParamsMinMaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  bool RunOnDevice() override {
    // Generate Int8 quant params based on the input data (last N samples of the
    // activations) and the quant scheme
    const float min =
          OperatorBase::Input<Tensor>(0, CPU).template data<float>()[0];
    const float max =
          OperatorBase::Input<Tensor>(1, CPU).template data<float>()[0];
    bool preserve_sparsity = false;
    if (InputSize() == 3){
        const auto* quant_scheme =
        this->template Input<unique_ptr<Int8QuantSchemeBlob>>(2).get();
        preserve_sparsity = quant_scheme->preserve_sparsity_;
    }
    dnnlowp::QuantizationFactory* qfactory =
        dnnlowp::QuantizationFactory::GetDefaultInstance();
    TensorQuantizationParams qparam = qfactory->ChooseQuantizationParams(
        min,
        max,
        8,
        preserve_sparsity);
    auto* output_qparam =
        this->template Output<unique_ptr<Int8QuantParamsBlob>>(0);
    output_qparam->reset(
        new Int8QuantParamsBlob(qparam.scale, qparam.zero_point));
    LOG_EVERY_N(INFO, 1) << "scale and bias are " << qparam.scale << "," << qparam.zero_point;
    return true;
  }

}; // class Int8GenQuantParamsOp

} // namespace caffe2
