#pragma once

#include "caffe2/predictor/predictor.h"
#include "caffe2/proto/caffe2_pb.h"

#include <memory>
#include <string>
#include <vector>

namespace caffe2 { namespace onnx {
class CAFFE2_API Caffe2BackendRep {
 public:
  void Run(
      const caffe2::Predictor::TensorList& inputs,
      caffe2::Predictor::TensorList* outputs);
  void RunMap(
      const caffe2::Predictor::TensorMap& inputs,
      caffe2::Predictor::TensorList* outputs);

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
  std::unique_ptr<caffe2::Predictor> predictor_{nullptr};
};
}}
