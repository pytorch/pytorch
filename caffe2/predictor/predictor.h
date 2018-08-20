#pragma once

#include <unordered_set>
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/predictor/predictor_config.h"
#include "caffe2/proto/metanet.pb.h"
#include "caffe2/proto/predictor_consts.pb.h"

namespace caffe2 {

class CAFFE2_API Predictor {
 public:
  using TensorVector = std::vector<TensorCPU*>;
  using TensorMap = std::unordered_map<std::string, TensorCPU*>;

  // MetaNetDef contains 'init_net', 'run_net', and meta-info
  // The meta-info is used to verify inputs are correctly passed
  Predictor(
      const MetaNetDef& net,
      Workspace* parent = nullptr,
      bool run_init = true);

  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`
  Predictor(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* parent = nullptr,
      bool run_init = true,
      int optimization = 1);

  ~Predictor() {}

  // Executes `run_net` on the inputs.
  // The first `inputs.size()` inputs from run_net::external_inputs
  // are shared with the data in `inputs`.

  // Precondition:
  //   inputs.size() <= run_net_.external_inputs.size()

  // Postcondition:
  //   outputs->size() == run_net.external_inputs.size()

  // Returns true on success
  bool run(const TensorVector& inputs, TensorVector* outputs);

  // Similar to run, but consumes a map of name to tensor as input
  bool run_map(const TensorMap& inputs, TensorVector* outputs);

  // Similar to the other run fns, except inputs and outputs are both maps of
  // string name to tensor.
  bool run_map_outputs(const TensorMap& inputs, TensorMap* outputs);

  const NetDef& def() const {
    return *config_.predict_net;
  };

  Workspace* ws() {
    return &ws_;
  };

  const std::vector<std::string>& input_names() const {
    return config_.input_names;
  }

  const std::vector<std::string>& output_names() const {
    return config_.output_names;
  }

 private:
  bool run_map_workspace(const TensorMap& inputs);
  PredictorConfig config_;
  Workspace ws_;
};
}
