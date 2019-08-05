#pragma once

#include "caffe2/core/workspace.h"

namespace caffe2 {

/**
 * This struct stores information about the inference graph which defines
 * underlying math of BlackBoxPredictor. Other parts of it such as various
 * threading optimizations don't belong here.
 */
struct InferenceGraph {
  std::unique_ptr<NetDef> predict_init_net_def;
  // shared_ptr allows to share NetDef with its operators on each of the threads
  // without memory replication. Note that predict_init_net_def_ could be stored
  // by value as its operators are discarded immidiatly after use (via
  // RunNetOnce)
  std::shared_ptr<NetDef> predict_net_def;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> parameter_names;
};
} // namespace caffe2
