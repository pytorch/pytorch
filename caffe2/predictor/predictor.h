#pragma once

#include <unordered_set>
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/predictor/predictor_config.h"

namespace caffe2 {

class TORCH_API Predictor {
 public:
  using TensorList = std::vector<TensorCPU>;
  using TensorMap = std::unordered_map<std::string, TensorCPU>;

  Predictor(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* parent = nullptr,
      bool run_init = true,
      int optimization = 1);

  Predictor(PredictorConfig config);

  virtual ~Predictor() {}

  // Executes `run_net` on the inputs.
  // The first `inputs.size()` inputs from run_net::external_inputs
  // are shared with the data in `inputs`.

  // Precondition:
  //   inputs.size() <= run_net_.external_inputs.size()

  // Postcondition:
  //   outputs->size() == run_net.external_inputs.size()

  // NOTE: output is a part of thread local workspace
  // and is only valid until the next predictor execution.

  // Returns true on success
  virtual bool operator()(const TensorList& inputs, TensorList* outputs);

  // Similar to run, but consumes a map of name to tensor as input
  bool operator()(const TensorMap& inputs, TensorList* outputs);

  // Similar to the other run fns, except inputs and outputs are both maps of
  // string name to tensor.
  bool operator()(const TensorMap& inputs, TensorMap* outputs);

  const NetDef& def() const {
    return *config_.predict_net;
  };

  Workspace* ws() {
    return config_.ws.get();
  };

  const std::vector<std::string>& input_names() const {
    return config_.input_names;
  }

  const std::vector<std::string>& output_names() const {
    return config_.output_names;
  }

 private:
  bool run_map_workspace(const TensorMap& inputs);

 protected:
  PredictorConfig config_;
};
} // namespace caffe2
