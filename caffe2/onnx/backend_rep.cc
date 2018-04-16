#include "caffe2/core/common.h"
#include "caffe2/onnx/backend_rep.h"
#include "caffe2/core/workspace.h"

#include <iostream>
#include <string>
#include <unordered_map>

namespace caffe2 { namespace onnx {

void Caffe2BackendRep::CheckInit() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!predictor_) {
    switch (pred_net_.device_option().device_type()) {
      case caffe2::DeviceType::CPU:
        predictor_ = caffe2::PredictorRegistry()->Create("CPU", init_net_, pred_net_, nullptr);
        break;
      case caffe2::DeviceType::CUDA:
        predictor_ = caffe2::PredictorRegistry()->Create("CUDA", init_net_, pred_net_, nullptr);
        break;
      default:
        CAFFE_THROW("Unsupported device type");
    }
    init_net_.Clear();
    pred_net_.Clear();
  }
}

void Caffe2BackendRep::Run(
    const TensorVector& inputs, OutputTensorVector* outputs) {
  CheckInit();
  predictor_->run(inputs, outputs);
}

void Caffe2BackendRep::RunMap(
    const TensorMap& inputs, OutputTensorVector* outputs) {
  CheckInit();
  predictor_->run_map(inputs, outputs);
}

}}
