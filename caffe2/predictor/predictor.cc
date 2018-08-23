#include "caffe2/predictor/predictor.h"
#include <unordered_set>
#include "caffe2/core/init.h"

namespace caffe2 {

namespace {

void enforceIsTensor(Workspace* ws, const std::string& name) {
  auto blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
  CAFFE_ENFORCE(
      blob->template IsType<Tensor>(CPU), "Blob is not a CPU Tensor: ", name);
}

void shareInputTensor(
    Workspace* ws,
    const std::string& name,
    TensorCPU* input) {
  enforceIsTensor(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  auto* tensor = blob->GetMutableTensor(CPU);
  tensor->ResizeLike(*input);
  tensor->ShareData(*input);
}

TensorCPU* extractOutputTensor(Workspace* ws, const std::string& name) {
  enforceIsTensor(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  return blob->GetMutableTensor(CPU);
}

} // namespace

Predictor::Predictor(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent,
    bool run_init,
    int optimization)
    : Predictor(makePredictorConfig(init_net, run_net, parent, run_init)) {}

Predictor::Predictor(PredictorConfig config) : config_(std::move(config)) {
  const auto& initialized_vec = config_.ws->Blobs();
  const std::unordered_set<std::string> initialized{initialized_vec.begin(),
                                                    initialized_vec.end()};
  for (const auto& name : config_.predict_net->external_input()) {
    if (!initialized.count(name)) {
      auto* blob = config_.ws->CreateBlob(name);
      blob->GetMutableTensor(CPU);
    }
  }
  CAFFE_ENFORCE(config_.ws->CreateNet(config_.predict_net));
}

bool Predictor::run(const TensorVector& inputs, TensorVector* outputs) {
  CAFFE_ENFORCE(
      inputs.size() <=
      static_cast<unsigned>(config_.predict_net->external_input_size()));
  for (size_t i = 0; i < inputs.size(); ++i) {
    shareInputTensor(
        config_.ws.get(), config_.predict_net->external_input(i), inputs[i]);
  }

  if (!config_.ws->RunNet(config_.predict_net->name())) {
    return false;
  }

  outputs->resize(config_.predict_net->external_output_size());
  for (size_t i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = extractOutputTensor(
        config_.ws.get(), config_.predict_net->external_output(i));
  }
  return true;
}

bool Predictor::run_map_workspace(const TensorMap& inputs) {
  if (!config_.input_names.empty()) {
    CAFFE_ENFORCE_EQ(inputs.size(), input_names().size());
  }
  for (auto input : inputs) {
    if (!input_names().empty()) {
      CAFFE_ENFORCE(
          std::find(input_names().begin(), input_names().end(), input.first) !=
              input_names().end(),
          "Input can't be found: ",
          input.first);
    }
    shareInputTensor(config_.ws.get(), input.first, input.second);
  }

  return config_.ws->RunNet(config_.predict_net->name());
}

bool Predictor::run_map(const TensorMap& inputs, TensorVector* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->resize(config_.predict_net->external_output_size());
  for (size_t i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = extractOutputTensor(
        config_.ws.get(), config_.predict_net->external_output(i));
  }
  return true;
}

bool Predictor::run_map_outputs(const TensorMap& inputs, TensorMap* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->reserve(output_names().size());
  for (const std::string& outputName : output_names()) {
    (*outputs)[outputName] = extractOutputTensor(config_.ws.get(), outputName);
  }
  return true;
}

} // namespace caffe2
