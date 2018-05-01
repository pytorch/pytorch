#include "caffe2/core/predictor.h"
#if CAFFE2_MOBILE
#include "caffe2/core/init.h"
#endif

#include <unordered_set>

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

// We don't use the getNet() from predictor_utils.cc here because that file
// has additional dependencies that we want to avoid bringing in, to keep the
// binary size as small as possible.
const NetDef& getNet(const MetaNetDef& def, const std::string& name) {
  for (const auto& n : def.nets()) {
    if (n.key() == name) {
      return n.value();
    }
  }
  CAFFE_THROW("Net not found: ", name);
}

const ::google::protobuf::RepeatedPtrField<::std::string>& getBlobs(
    const MetaNetDef& def,
    const std::string& name) {
  for (const auto& b : def.blobs()) {
    if (b.key() == name) {
      return b.value();
    }
  }
  CAFFE_THROW("Blob not found: ", name);
}
} // namespace

Predictor::Predictor(const MetaNetDef& def, Workspace* parent, bool run_init)
    : Predictor(
          getNet(
              def,
              PredictorConsts::default_instance().global_init_net_type()),
          getNet(def, PredictorConsts::default_instance().predict_net_type()),
          parent,
          run_init) {
  const auto& inputs =
      getBlobs(def, PredictorConsts::default_instance().inputs_blob_type());
  for (const auto& input : inputs) {
    inputNames_.insert(input);
  }

  const auto& outputs =
      getBlobs(def, PredictorConsts::default_instance().outputs_blob_type());
  for (const auto& output : outputs) {
    outputNames_.emplace_back(output);
  }
}

Predictor::Predictor(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent,
    bool run_init)
    : run_net_(run_net), ws_(parent) {
  if (run_init) {
    CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  }
#if CAFFE2_MOBILE
  GlobalInit();
#endif

  // real model inputs can be fed later in run* functions
  const auto& initialized_vec = ws_.Blobs();
  const std::unordered_set<std::string> initialized{initialized_vec.begin(),
                                                    initialized_vec.end()};
  for (const auto& name : run_net.external_input()) {
    if (!initialized.count(name)) {
      auto* blob = ws_.CreateBlob(name);
      blob->template GetMutable<TensorCPU>();
    }
  }
  CAFFE_ENFORCE(ws_.CreateNet(run_net));
}

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

bool Predictor::run_map_workspace(const TensorMap& inputs) {
  if (!inputNames_.empty()) {
    CAFFE_ENFORCE_EQ(inputs.size(), inputNames_.size());
  }
  for (auto input : inputs) {
    if (!inputNames_.empty()) {
      CAFFE_ENFORCE_GT(inputNames_.count(input.first), 0);
    }
    shareInputTensor(&ws_, input.first, input.second);
  }

  return ws_.RunNet(run_net_.name());
}

bool Predictor::run_map(const TensorMap& inputs, TensorVector* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->resize(run_net_.external_output_size());
  for (auto i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = extractOutputTensor(&ws_, run_net_.external_output(i));
  }
  return true;
}

bool Predictor::run_map_outputs(const TensorMap& inputs, TensorMap* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->reserve(outputNames_.size());
  for (const std::string& outputName : outputNames_) {
    (*outputs)[outputName] = extractOutputTensor(&ws_, outputName);
  }
  return true;
}

} // namespace caffe2
