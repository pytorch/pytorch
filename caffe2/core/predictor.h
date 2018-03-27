#pragma once

#include <unordered_set>
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/metanet.pb.h"
#include "caffe2/proto/predictor_consts.pb.h"

namespace caffe2 {

class Predictor {
 public:
  using TensorVector = std::vector<TensorCPU*>;
  using TensorMap = std::unordered_map<std::string, TensorCPU*>;

  // MetaNetDef contains 'init_net', 'run_net', and meta-info
  // The meta-info is used to verify inputs are correctly passed
  Predictor(const MetaNetDef& net, Workspace* parent = nullptr);

  // Runs the `init_net` once, then saves the `run_net` to be executed
  // in `::run`
  Predictor(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* parent = nullptr);
  ~Predictor();

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

  const NetDef& def() const {
    return run_net_;
  };

  Workspace* ws() {
    return &ws_;
  };

 private:
  NetDef run_net_;
  Workspace ws_;
  std::unordered_set<std::string> inputNames_;
};
}
