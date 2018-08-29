#include "predictor_config.h"
#include "caffe2/core/init.h"
#ifdef CAFFE2_OPTIMIZER
#include "caffe2/opt/optimizer.h"
#endif
namespace caffe2 {

namespace {

// We don't use the getNet() from predictor_utils.cc here because that file
// has additional dependencies that we want to avoid bringing in, to keep the
// binary size as small as possible.
static const NetDef& getNet(const MetaNetDef& def, const std::string& name) {
  for (const auto& n : def.nets()) {
    if (n.key() == name) {
      return n.value();
    }
  }
  CAFFE_THROW("Net not found: ", name);
}

static const ::google::protobuf::RepeatedPtrField<::std::string>& getBlobs(
    const MetaNetDef& def,
    const std::string& name) {
  for (const auto& b : def.blobs()) {
    if (b.key() == name) {
      return b.value();
    }
  }
  CAFFE_THROW("Blob not found: ", name);
}

static std::string combine(const std::string& str, const std::string& name) {
  if (name.empty()) {
    return std::string(str);
  }
  return str + "_" + name;
}

static std::string getNamedPredictNet(const string& name) {
  return combine(PredictorConsts::default_instance().predict_net_type(), name);
}

static std::string getNamedInitNet(const string& name) {
  return combine(
      PredictorConsts::default_instance().predict_init_net_type(), name);
}

static std::string getNamedInputs(const string& name) {
  return combine(PredictorConsts::default_instance().inputs_blob_type(), name);
}

static std::string getNamedOutputs(const string& name) {
  return combine(PredictorConsts::default_instance().outputs_blob_type(), name);
}

static std::string getNamedParams(const string& name) {
  return combine(
      PredictorConsts::default_instance().parameters_blob_type(), name);
}

} // namespace

PredictorConfig makePredictorConfig(
    const MetaNetDef& def,
    Workspace* parent,
    bool run_init,
    const std::string& net_name) {
  const auto& init_net = getNet(def, getNamedInitNet(net_name));
  const auto& run_net = getNet(def, getNamedPredictNet(net_name));
  auto config = makePredictorConfig(init_net, run_net, parent, run_init);
  const auto& inputs = getBlobs(def, getNamedInputs(net_name));
  for (const auto& input : inputs) {
    config.input_names.emplace_back(input);
  }

  const auto& outputs = getBlobs(def, getNamedOutputs(net_name));
  for (const auto& output : outputs) {
    config.output_names.emplace_back(output);
  }

  const auto& params = getBlobs(def, getNamedParams(net_name));
  for (const auto& param : params) {
    config.parameter_names.emplace_back(param);
  }

  return config;
}

PredictorConfig makePredictorConfig(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent,
    bool run_init,
    int optimization) {
  PredictorConfig config;
  config.ws = std::make_shared<Workspace>(parent);
  auto& ws = *config.ws;
  config.predict_net = std::make_shared<NetDef>(run_net);
  if (run_init) {
    CAFFE_ENFORCE(ws.RunNetOnce(init_net));
  }
#if CAFFE2_MOBILE
  GlobalInit();
#endif
  if (optimization) {
#ifdef CAFFE2_OPTIMIZER
    try {
      *config.predict_net =
          opt::optimize(*config.predict_net, &ws, optimization);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Optimization pass failed: " << e.what();
    }
#else
    LOG(WARNING) << "Caffe2 is compiled without optimization passes.";
#endif
  }
  return config;
}

} // namespace caffe2
