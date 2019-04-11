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

#include "caffe2/core/operator.h"
#include "caffe2/intra_op_parallel/intra_op_parallel.h"
#include "caffe2/sgd/adagrad_op.h"

namespace caffe2 {

namespace intra_op_parallel {

template <typename T>
class ParallelAdagradOp final : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  ParallelAdagradOp(const OperatorDef& operator_def, Workspace* ws);

 protected:
  bool RunOnDevicePrologue(int /* unused */) override;
  bool RunOnDeviceParallel(int task_id, int num_tasks) override;

  const T epsilon_;
  const T decay_;
  INPUT_TAGS(PARAM, MOMENT_1, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T>
class ParallelSparseAdagradOp final : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DISPATCH_HELPER;
  ParallelSparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : ParallelOpBase(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}

 protected:
  bool RunOnDevicePrologue(int /* unused */) override;
  bool RunOnDeviceParallel(int task_id, int num_tasks) override;
  template <typename SIndex>
  bool DoRunWithType(int task_id, int num_tasks);

  const T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

} // namespace intra_op_parallel

} // namespace caffe2
