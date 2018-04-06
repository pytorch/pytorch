#pragma once

#include <unordered_set>
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/metanet.pb.h"
#include "caffe2/proto/predictor_consts.pb.h"

namespace caffe2 {

class PredictorBase;

CAFFE_DECLARE_REGISTRY(
  PredictorRegistry,
  PredictorBase,
  const NetDef &,
  const NetDef &,
  Workspace *);

#define CAFFE_REGISTER_PREDICTOR(key, clsname) \
  CAFFE_REGISTER_CLASS(PredictorRegistry, key, clsname)

#define USE_PREDICTOR_CONTEXT_FUNCTIONS \
  using TensorVector = PredictorBase::TensorVector; \
  using TensorMap = PredictorBase::TensorMap; \
  using OutputTensorVector = PredictorBase::OutputTensorVector

namespace predictor_details {

const NetDef& getNet(const MetaNetDef& def, const std::string& name);

template <class TensorType>
void enforceIsTensor(Workspace* ws, const std::string& name) {
  auto blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
  CAFFE_ENFORCE(
      blob->template IsType<TensorType>(), "Blob is not a Tensor of the required type ", name);
}

} // namespace predictor_details

class PredictorBase {
 public:
  using TensorVector = std::vector<TensorCPU*>;
  using TensorMap = std::unordered_map<std::string, TensorCPU*>;
  using OutputTensorVector = std::vector<std::shared_ptr<TensorCPU>>;

  // MetaNetDef contains 'init_net', 'run_net', and meta-info
  // The meta-info is used to verify inputs are correctly passed
  PredictorBase(const MetaNetDef& net, Workspace* parent = nullptr);

  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`
  PredictorBase(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* parent = nullptr);

  ~PredictorBase();

  // Executes `run_net` on the inputs.
  // The first `inputs.size()` inputs from run_net::external_inputs
  // are shared with the data in `inputs`.

  // Precondition:
  //   inputs.size() <= run_net_.external_inputs.size()

  // Postcondition:
  //   outputs->size() == run_net.external_inputs.size()

  // Returns true on success
  virtual bool run(const TensorVector& inputs, OutputTensorVector& outputs, bool mulithread) = 0;

  // Similar to run, but consumes a map of name to tensor as input
  virtual bool run_map(const TensorMap& inputs, OutputTensorVector& outputs, bool mulithread) = 0;

  const NetDef& def() const {
    return run_net_;
  };

  Workspace* ws() {
    return &ws_;
  };

 protected:
  std::unordered_set<std::string> initialized_;
  NetDef run_net_;
  Workspace ws_;
  std::unordered_set<std::string> inputNames_;
};

template <class Context>
class Predictor : public PredictorBase {
 public:
  USE_PREDICTOR_CONTEXT_FUNCTIONS;

  Predictor(const MetaNetDef& def, Workspace* parent = nullptr)
  : Predictor(
      predictor_details::getNet(def, PredictorConsts::default_instance().global_init_net_type()),
      predictor_details::getNet(def, PredictorConsts::default_instance().predict_net_type()),
      parent) {
    const auto& inputs =
        predictor_details::getBlobs(def, PredictorConsts::default_instance().inputs_blob_type());
    for (const auto& input : inputs) {
      inputNames_.insert(input);
    }
  }

  Predictor(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* parent = nullptr)
    : PredictorBase(init_net, run_net, parent)
  {
    context_ = std::make_shared<Context>(run_net.device_option());
    for (const auto& name : run_net.external_input()) {
      if (!initialized_.count(name)) {
        auto* blob = ws_.CreateBlob(name);
        CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
        blob->template GetMutable<Tensor<Context>>();
      }
    }
    CAFFE_ENFORCE(ws_.CreateNet(run_net));
  }

  virtual bool run(const TensorVector& inputs, OutputTensorVector& outputs, bool threadsafe = false) override;
  virtual bool run_map(const TensorMap& inputs, OutputTensorVector& outputs, bool threadsafe = false) override;

 private:
  std::shared_ptr<Context> context_;
};

template <class Context>
bool Predictor<Context>::run(const TensorVector& inputs, OutputTensorVector& outputs, bool threadsafe) {
  TensorMap input_map;
  CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
  for (auto i = 0; i < inputs.size(); ++i) {
    input_map.emplace(run_net_.external_input(i), inputs[i]);
  }

  return run_map(input_map, outputs);
}
}
