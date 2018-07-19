#include "caffe2/core/common.h"
#include "caffe2/onnx/backend_rep.h"

#include <iostream>

namespace caffe2 { namespace onnx {

void Caffe2BackendRep::CheckInit() {
  if (!predictor_) {
    predictor_ = caffe2::make_unique<caffe2::Predictor>(init_net_, pred_net_);
    init_net_.Clear();
    pred_net_.Clear();
  }
}


void Caffe2BackendRep::Run(
    const caffe2::Predictor::TensorVector& inputs,
    caffe2::Predictor::TensorVector* outputs) {
  CheckInit();
  predictor_->run(inputs, outputs);
}

void Caffe2BackendRep::RunMap(
    const caffe2::Predictor::TensorMap& inputs,
    caffe2::Predictor::TensorVector* outputs) {
  CheckInit();
  predictor_->run_map(inputs, outputs);
}

}}
