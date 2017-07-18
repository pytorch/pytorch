#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

class RecurrentNetworkExecutor {
 public:
  explicit RecurrentNetworkExecutor(const NetDef& step_net_def)
      : step_net_def_(step_net_def) {}

  ~RecurrentNetworkExecutor() {}

  bool RunTimestep(int t, Workspace* ws) {
    EnsureTimestepInitialized(t, ws);
    Exec(t);
    return true;
  }

 private:
  void Exec(int t) {
    for (auto& op : timestep_ops_[t]) {
      bool success = op->RunAsync();
      CAFFE_ENFORCE(
          success, "Failure running: " + ProtoDebugString(op->debug_def()));
    }
  }

  void EnsureTimestepInitialized(int t, Workspace* ws) {
    if (timestep_ops_.size() <= t) {
      CAFFE_ENFORCE_EQ(
          timestep_ops_.size(), t, "You must call timestaps sequentially");
      timestep_ops_.push_back(std::vector<unique_ptr<OperatorBase>>());

      for (const OperatorDef& operator_def : step_net_def_.op()) {
        timestep_ops_.back().emplace_back(CreateOperator(operator_def, ws));
      }
    }
  }

  std::vector<std::vector<unique_ptr<OperatorBase>>> timestep_ops_;
  NetDef step_net_def_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_
