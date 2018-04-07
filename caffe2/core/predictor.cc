#include "caffe2/core/predictor.h"

#include <unordered_map>

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
  PredictorRegistry,
  PredictorBase,
  const NetDef &,
  const NetDef &,
  Workspace *);


namespace predictor_details {

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

void shareInputTensor(
    Workspace* ws,
    const std::string& name,
    TensorCPU* input) {
  enforceIsTensor<TensorCPU>(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  auto* tensor = blob->template GetMutable<TensorCPU>();
  tensor->ResizeLike(*input);
  tensor->ShareData(*input);
}

std::shared_ptr<TensorCPU> extractOutputTensor(Workspace* ws, const std::string& name) {
  enforceIsTensor<TensorCPU>(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");

  // Since the tensor is a member of blob, we should not delete the pointer.
  // By given a customized deleter, the pointer won't be deleted on shared_ptr deconstruction.
  return std::shared_ptr<TensorCPU>(blob->template GetMutable<TensorCPU>(), [](TensorCPU *){});
}

} // namespace predictor_details

PredictorBase::PredictorBase(const MetaNetDef& def, Workspace* parent)
  : PredictorBase(
          predictor_details::getNet(
              def,
              PredictorConsts::default_instance().global_init_net_type()),
          predictor_details::getNet(def, PredictorConsts::default_instance().predict_net_type()),
          parent) {
  const auto& inputs =
      predictor_details::getBlobs(def, PredictorConsts::default_instance().inputs_blob_type());
  for (const auto& input : inputs) {
    inputNames_.insert(input);
  }
}

PredictorBase::PredictorBase(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent)
: run_net_(run_net), ws_(parent) {
  CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  // real model inputs can be fed later in run* functions

  const auto& initialized_vec = ws_.Blobs();
  initialized_ = std::unordered_set<std::string>{initialized_vec.begin(),
                                                 initialized_vec.end()};
  for (const auto& name : run_net.external_input()) {
    if (!initialized_.count(name)) {
      auto* blob = ws_.CreateBlob(name);
    }
  }
  CAFFE_ENFORCE(ws_.CreateNet(run_net));
}

PredictorBase::~PredictorBase() {}

template <>
bool Predictor<CPUContext>::run_map(const TensorMap& inputs, OutputTensorVector& outputs, bool threadsafe) {
  if (!inputNames_.empty()) {
    CAFFE_ENFORCE_EQ(inputs.size(), inputNames_.size());
  }
  if (!threadsafe) {
    // If threadsafe is false, we run the net in the contained workspace of the predictor
    for (auto &input : inputs) {
      if (!inputNames_.empty()) {
        CAFFE_ENFORCE_GT(inputNames_.count(input.first), 0);
      }
      predictor_details::shareInputTensor(&ws_, input.first, input.second);
    }

    if (!ws_.RunNet(run_net_.name())) {
      return false;
    }

    outputs.resize(run_net_.external_output_size());
    for (auto i = 0; i < outputs.size(); ++i) {
      outputs[i] = predictor_details::extractOutputTensor(&ws_, run_net_.external_output(i));
    }
  } else {
    // If threadsafe is true, we create a temporary new workspace and load the network to the new workspace

    // Instead of running the init_net in the new workspace
    // We directly forward these blobs from ws_ for efficiency.
    // Note: RNN might have problem doing this way.
    std::unordered_map<std::string, std::string> forwarded_blobs;
    for (auto &input : initialized_) {
      forwarded_blobs.emplace(input, input);
    }
    Workspace ws(&ws_, forwarded_blobs);

    // For each input, create a local blob without touching ws_.
    // This will override the mapped blob from ws_.
    for (auto &input: inputs) {
      if (!inputNames_.empty()) {
        CAFFE_ENFORCE_GT(inputNames_.count(input.first), 0);
      }
      auto *blob = ws.CreateLocalBlob(input.first);
      CAFFE_ENFORCE(blob, "Blob: ", input.first, " does not exist");
      blob->template GetMutable<TensorCPU>();
      predictor_details::shareInputTensor(&ws, input.first, input.second);
    }

    if (!ws.RunNetOnce(run_net_)) {
      return false;
    }

    outputs.resize(run_net_.external_output_size());
    for (auto i = 0; i < outputs.size(); ++i) {
      auto name = run_net_.external_output(i);
      auto *blob = ws.GetBlob(name);
      CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
      outputs[i] = std::make_shared<TensorCPU>();
      outputs[i]->CopyFrom(*blob->template GetMutable<TensorCPU>(), context_.get());
    }
  }
  return true;
}

CAFFE_REGISTER_PREDICTOR(CPU, Predictor<CPUContext>);
} // namespace caffe2
