#include "caffe2/predictor/predictor.h"
#ifdef CAFFE2_OPTIMIZER
#include "caffe2/opt/optimizer.h"
#endif
#include "caffe2/utils/proto_utils.h"

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
    config_.input_names.emplace_back(input);
  }

  const auto& outputs =
      getBlobs(def, PredictorConsts::default_instance().outputs_blob_type());
  for (const auto& output : outputs) {
    config_.output_names.emplace_back(output);
  }
}

Predictor::Predictor(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent,
    bool run_init,
    int optimization)
    : ws_(parent) {
  config_.predict_net = std::make_shared<NetDef>(run_net);
  if (run_init) {
    CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  }
#if CAFFE2_MOBILE
  GlobalInit();
#endif
  auto predict_net = config_.predict_net;

  if (optimization &&
      !ArgumentHelper::HasArgument(*predict_net, "disable_nomnigraph")) {
#ifdef CAFFE2_OPTIMIZER
    try {
      *predict_net = opt::optimize(*predict_net, &ws_, optimization);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Optimization pass failed: " << e.what();
    }
#else
    LOG(WARNING) << "Caffe2 is compiled without optimization passes.";
#endif
  }

  // real model inputs can be fed later in run* functions
  const auto& initialized_vec = ws_.Blobs();
  const std::unordered_set<std::string> initialized{initialized_vec.begin(),
                                                    initialized_vec.end()};
  for (const auto& name : predict_net->external_input()) {
    if (!initialized.count(name)) {
      auto* blob = ws_.CreateBlob(name);
      blob->GetMutableTensor(CPU);
    }
  }

  CAFFE_ENFORCE(ws_.CreateNet(predict_net));
}

bool Predictor::run(const TensorVector& inputs, TensorVector* outputs) {
  CAFFE_ENFORCE(
      inputs.size() <=
      static_cast<unsigned>(config_.predict_net->external_input_size()));
  for (size_t i = 0; i < inputs.size(); ++i) {
    shareInputTensor(&ws_, config_.predict_net->external_input(i), inputs[i]);
  }

  if (!ws_.RunNet(config_.predict_net->name())) {
    return false;
  }

  outputs->resize(config_.predict_net->external_output_size());
  for (size_t i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] =
        extractOutputTensor(&ws_, config_.predict_net->external_output(i));
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
    shareInputTensor(&ws_, input.first, input.second);
  }

  return ws_.RunNet(config_.predict_net->name());
}

bool Predictor::run_map(const TensorMap& inputs, TensorVector* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->resize(config_.predict_net->external_output_size());
  for (size_t i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] =
        extractOutputTensor(&ws_, config_.predict_net->external_output(i));
  }
  return true;
}

bool Predictor::run_map_outputs(const TensorMap& inputs, TensorMap* outputs) {
  if (!run_map_workspace(inputs)) {
    return false;
  }

  outputs->reserve(output_names().size());
  for (const std::string& outputName : output_names()) {
    (*outputs)[outputName] = extractOutputTensor(&ws_, outputName);
  }
  return true;
}

} // namespace caffe2
