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

namespace caffe2 {

namespace intra_op_parallel {

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
class ParallelFullyConnectedOp final : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  ParallelFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws);
  ~ParallelFullyConnectedOp() override {}

 private:
  bool RunOnDevicePrologue(int num_tasks) override;
  bool RunOnDeviceParallel(int task_id, int num_tasks) override;

  size_t axis_{1};
  size_t axis_w_{1};
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;

  struct Partition2D {
    int num_tasks{0}, M, N;
    int m_begin, m_end, n_begin, n_end;
    int padding[9];
  };

  std::vector<Partition2D> partition_cache_;
};

/**
 * FCGradient needs 2 sets of parallel tasks and each set wants to use
 * different partitioning for better performance, which does not quite fit
 * the RunOnDeviceParallel interface of ParallelOpBase.
 */
class ParallelFullyConnectedGradientOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  ParallelFullyConnectedGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws);
  ~ParallelFullyConnectedGradientOp() override {}

 private:
  bool RunOnDevice() final;
  bool HasAsyncPart() const override {
    return true;
  }

  bool RunOnDevicePrologue();
  bool RunOnDeviceParallel(int task_id, int num_tasks);
  bool ComputeDWParallel_(int task_id, int num_tasks);
  void ComputeDXParallel_(int task_id, int num_tasks);

  size_t axis_{1};
  size_t axis_w_{1};

  int max_num_tasks_;
  std::atomic<int> count_;
};

} // namespace intra_op_parallel

} // namespace caffe2
