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

#include <tbb/partitioner.h>

namespace caffe2 {

namespace tbb {

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
class TBBFullyConnectedOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  TBBFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws);
  ~TBBFullyConnectedOp() {}

 private:
  bool RunOnDevice() final;

  bool RunOnDevicePrologue(int num_tasks);
  bool RunOnDeviceParallel(int task_id, int num_tasks);

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
  int max_num_workers_;
  ::tbb::affinity_partitioner ap;
};

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

  int max_num_workers_;
  ::tbb::affinity_partitioner ap;
};

} // namespace tbb

} // namespace caffe2
