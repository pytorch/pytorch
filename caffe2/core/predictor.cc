#include "caffe2/core/predictor.h"

namespace caffe2 {

namespace {

void enforceIsTensor(Workspace* ws, const std::string& name) {
  auto blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
  CAFFE_ENFORCE(
      blob->template IsType<TensorCPU>(), "Blob is not a CPU Tensor: ", name);
}

void shareInputTensor(
    Workspace* ws,
    const std::string& name,
    TensorCPU* input) {
  enforceIsTensor(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  auto* tensor = blob->template GetMutable<TensorCPU>();
  tensor->ResizeLike(*input);
  tensor->ShareData(*input);
}

TensorCPU* extractOutputTensor(Workspace* ws, const std::string& name) {
  enforceIsTensor(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  return blob->template GetMutable<TensorCPU>();
}
}

Predictor::Predictor(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent)
    : run_net_(run_net), ws_(parent) {
  CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  CAFFE_ENFORCE(ws_.CreateNet(run_net));
}

Predictor::~Predictor() {}

bool Predictor::run(const TensorVector& inputs, TensorVector* outputs) {
  CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
  for (auto i = 0; i < inputs.size(); ++i) {
    shareInputTensor(&ws_, run_net_.external_input(i), inputs[i]);
  }

  if (!ws_.RunNet(run_net_.name())) {
    return false;
  }

  outputs->resize(run_net_.external_output_size());
  for (auto i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = extractOutputTensor(&ws_, run_net_.external_output(i));
  }
  return true;
}
}
