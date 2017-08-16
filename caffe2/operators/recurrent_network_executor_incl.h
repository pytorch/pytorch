// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_INCL_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_INCL_H_

#include <vector>
#include "caffe2/core/operator.h"

namespace caffe2 {

struct RNNNetOperator {
  // Operator
  int order;
  std::shared_ptr<OperatorBase> op = nullptr;
  bool link_op;

  // Bookkeeping
  int num_dynamic_inputs = 0;
  int num_recurrent_inputs = 0;

  // Dependencies
  std::atomic<int> proc_inputs;
  std::vector<int> dependencies;
  std::vector<int> parents;
  bool frontier = true;

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

struct OpJob {
  int timestep;
  int op_idx;
  int T;
  int direction;
  int stream_id = -1; // only used by gpu version
  OpJob() {}
  OpJob(int _timestep, int _op_idx, int _T, int _direction)
      : timestep(_timestep), op_idx(_op_idx), T(_T), direction(_direction) {
    CHECK(direction == 1 || direction == -1);
    CHECK(timestep >= 0 && timestep < _T);
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
