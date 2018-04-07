#pragma once

#include "caffe2/core/predictor.h"
#include "caffe2/proto/caffe2.pb.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caffe2 { namespace onnx {
class Caffe2BackendRep {
 public:
  using TensorVector = caffe2::PredictorBase::TensorVector;
  using TensorMap = caffe2::PredictorBase::TensorMap;
  using OutputTensorVector = caffe2::PredictorBase::OutputTensorVector;

  void Run(const TensorVector& inputs, OutputTensorVector* outputs, bool threadsafe = false);
  void RunMap(const TensorMap& inputs, OutputTensorVector* outputs, bool threadsafe = false);

  caffe2::NetDef& init_net() {
    return init_net_;
  }
  caffe2::NetDef& pred_net() {
    return pred_net_;
  }
  std::vector<std::string>& uninitialized_inputs() {
    return uninitialized_inputs_;
  }

  const caffe2::NetDef& init_net() const {
    return init_net_;
  }
  const caffe2::NetDef& pred_net() const {
    return pred_net_;
  }
  const std::vector<std::string>& uninitialized_inputs() const {
    return uninitialized_inputs_;
  }

 private:
  void CheckInit();

  caffe2::NetDef init_net_;
  caffe2::NetDef pred_net_;
  std::vector<std::string> uninitialized_inputs_;
  std::unique_ptr<caffe2::PredictorBase> predictor_{nullptr};
  std::mutex mutex_;
};
}}
