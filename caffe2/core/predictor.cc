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

const NetDef& GetNet(const MetaNetDef& def, const std::string& name) {
  for (const auto& n : def.nets()) {
    if (n.key() == name) {
      return n.value();
    }
  }
  CAFFE_THROW("Net not found: ", name);
}

const ::google::protobuf::RepeatedPtrField<::std::string>& GetBlobs(
    const MetaNetDef& def,
    const std::string& name) {
  for (const auto& b : def.blobs()) {
    if (b.key() == name) {
      return b.value();
    }
  }
  CAFFE_THROW("Blob not found: ", name);
}

void ShareInputTensor(
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

std::shared_ptr<TensorCPU> ExtractOutputTensor(Workspace* ws, const std::string& name) {
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
          predictor_details::GetNet(
              def,
              PredictorConsts::default_instance().global_init_net_type()),
          predictor_details::GetNet(def, PredictorConsts::default_instance().predict_net_type()),
          parent) {
  const auto& inputs =
      predictor_details::GetBlobs(def, PredictorConsts::default_instance().inputs_blob_type());
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
}

PredictorBase::~PredictorBase() {}

template <>
bool Predictor<CPUContext>::run_map(const TensorMap& inputs, OutputTensorVector* outputs) {
  if (!inputNames_.empty()) {
    CAFFE_ENFORCE_EQ(inputs.size(), inputNames_.size());
  }
  for (auto &input : inputs) {
    if (!inputNames_.empty()) {
      CAFFE_ENFORCE_GT(inputNames_.count(input.first), 0);
    }
    predictor_details::ShareInputTensor(&ws_, input.first, input.second);
  }

  if (!ws_.RunNet(run_net_.name())) {
    return false;
  }

  outputs->resize(run_net_.external_output_size());
  for (auto i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = predictor_details::ExtractOutputTensor(&ws_, run_net_.external_output(i));
  }
  return true;
}

CAFFE_REGISTER_PREDICTOR(CPU, Predictor<CPUContext>);
} // namespace caffe2
