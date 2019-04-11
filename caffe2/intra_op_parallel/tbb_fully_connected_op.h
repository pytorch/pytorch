/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

#include "caffe2/intra_op_parallel/intra_op_parallel.h"

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB

namespace caffe2 {

namespace tbb {

/**
 * We could reuse ParallelFullyConnectedGradientOp in
 * parallel_fully_connected_op.h but this implementation shows using TBB
 * is easier to implement an operator with a complex intra-op parallelism
 */
class TBBFullyConnectedGradientOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  TBBFullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws);
  ~TBBFullyConnectedGradientOp() {}

 private:
  bool RunOnDevice() final;

  bool RunOnDevicePrologue();
  bool ComputeDWParallel_(int task_id, int num_tasks);
  void ComputeDXParallel_(int task_id, int num_tasks);

  size_t axis_{1};
  size_t axis_w_{1};

  int max_num_tasks_;
};

} // namespace tbb

} // namespace caffe2

#endif // INTRA_OP_PARALLEL_CAN_USE_TBB
