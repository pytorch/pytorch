
#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_INCL_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_INCL_H_

#include <vector>
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 * Struct for operator in a timestep and its dependencies.
 */
struct RNNNetOperator {
  int order; // Position in the step net (i.e nth operator)
  std::shared_ptr<OperatorBase> op = nullptr;
  bool link_op; // Special flag for link op, see RNNApplyLinkOp.

  // Bookkeeping, used by ThreadedRecurrentNetworkExecutor
  int num_dynamic_inputs = 0;
  int num_recurrent_inputs = 0;
  std::atomic<int> proc_inputs;

  // Dependencies to other ops. If dependency index < order, it is
  // a recurrent dependency (i.e to the next timestep)
  std::vector<int> dependencies;
  std::vector<int> parents;
  bool frontier = true; // For ops that are launched first
  bool has_timestep_blob = false;

  explicit RNNNetOperator(const OperatorDef& def, int order) : order(order) {
    proc_inputs = 0;
    link_op = def.type() == "rnn_internal_apply_link";
  }

  RNNNetOperator(const RNNNetOperator& x) {
    order = x.order;
    op = x.op;
    link_op = x.link_op;
    num_dynamic_inputs = x.num_dynamic_inputs;
    num_recurrent_inputs = x.num_recurrent_inputs;
    proc_inputs = 0;
    dependencies = x.dependencies;
    parents = x.parents;
    frontier = x.frontier;
  }
};

/**
 * Data structure for a scheduled task in the task queue.
 */
struct OpTask {
  int timestep;
  int op_idx; // matches RNNNetOperator.order
  int T; // number of timesteps in this execution
  int direction; // +1 for forward, -1 for backward pass
  int stream_id = -1; // only used by gpu version
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  OpTask() {}
  OpTask(int _timestep, int _op_idx, int _T, int _direction)
      : timestep(_timestep), op_idx(_op_idx), T(_T), direction(_direction) {
    CAFFE_ENFORCE(direction == 1 || direction == -1);
    CAFFE_ENFORCE(timestep >= 0 && timestep < _T);
  }

  inline bool backward() {
    return direction == -1;
  }
  inline bool forward() {
    return direction == 1;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_
