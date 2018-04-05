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

void Caffe2BackendRep::RunInNewWorkspace(
    const caffe2::Predictor::TensorVector& inputs,
    std::vector<std::shared_ptr<caffe2::TensorCPU>> &outputs) {
  CheckInit();

  auto &run_net = predictor_->def();

  // Mapping all external input blobs from predictor workspace to the new workspace.
  // This includes both pretrained weights and network input blobs.
  std::unordered_map<std::string, std::string> forwarded_blobs;
  for (auto &input : run_net.external_input()) {
    forwarded_blobs.emplace(input, input);
  }
  Workspace ws(predictor_->ws(), forwarded_blobs);

  // For each network input, create a local blob.
  // This will override the mapped blob from predictor workspace.
  for (auto i = 0; i < inputs.size(); ++i) {
    auto &name = run_net.external_input(i);
    auto blob = ws.CreateLocalBlob(name);
    auto tensor = blob->template GetMutable<TensorCPU>();
    auto input = inputs[i];
    tensor->ResizeLike(*input);
    tensor->ShareData(*input);
  }

  ws.RunNetOnce(run_net);

  outputs.resize(run_net.external_output_size());
  for (auto i = 0; i < outputs.size(); ++i) {
    auto &name = run_net.external_output(i);
    auto blob = ws.GetBlob(name);
    outputs[i].reset(new TensorCPU());
    outputs[i]->CopyFrom(*blob->template GetMutable<TensorCPU>());
  }
}

void Caffe2BackendRep::RunMapInNewWorkspace(
    const caffe2::Predictor::TensorMap& inputs,
    std::vector<std::shared_ptr<caffe2::TensorCPU>> &outputs) {
  CheckInit();

  auto &run_net = predictor_->def();

  // Mapping all external input blobs from predictor workspace to the new workspace.
  // This includes both pretrained weights and network input blobs.
  std::unordered_map<std::string, std::string> forwarded_blobs;
  for (auto &input : run_net.external_input()) {
    forwarded_blobs.emplace(input, input);
  }
  Workspace ws(predictor_->ws(), forwarded_blobs);

  // For each network input, create a local blob.
  // This will override the mapped blob from predictor workspace.
  for (auto &input: inputs) {
    auto &name = input.first;
    auto &input_tensor = input.second;
    auto blob = ws.CreateLocalBlob(name);
    auto tensor = blob->template GetMutable<TensorCPU>();
    tensor->ResizeLike(*input_tensor);
    tensor->ShareData(*input_tensor);
  }

  ws.RunNetOnce(run_net);

  outputs.resize(run_net.external_output_size());
  for (auto i = 0; i < outputs.size(); ++i) {
    auto &name = run_net.external_output(i);
    auto blob = ws.GetBlob(name);
    outputs[i].reset(new TensorCPU());
    outputs[i]->CopyFrom(*blob->template GetMutable<TensorCPU>());
  }
}

}}
