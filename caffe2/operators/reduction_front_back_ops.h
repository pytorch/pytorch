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

#ifndef CAFFE2_OPERATORS_REDUCTION_FRONT_BACK_OPS_H_
#define CAFFE2_OPERATORS_REDUCTION_FRONT_BACK_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context, bool FIRSTDIMS>
class MaxReduceDimsOp final : public Operator<Context> {
 public:
  MaxReduceDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() {
    auto& X = Input(0);
    auto* Y = Output(0);

    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.ndim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.ndim() - num_reduce_dims_);

    vector<TIndex> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index =
        FIRSTDIMS ? X.dims().size() : X.dims().size() - num_reduce_dims_;

    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(X.dims()[i]);
    }
    Y->Resize(output_shape);

    if (cols == 0 || rows == 0) {
      return true;
    }

    const float* data = X.template data<float>();
    float* out_data = Y->template mutable_data<float>();
    Compute(rows, cols, data, out_data);
    return true;
  }

 protected:
  void Compute(int rows, int cols, const float* data, float* out_data);

  int num_reduce_dims_;
};

template <typename T, class Context, bool FIRSTDIMS>
class MaxReduceDimsGradientOp final : public Operator<Context> {
 public:
  MaxReduceDimsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& dY = Input(0);
    auto& X = Input(1);
    auto& Y = Input(2);
    auto* dX = Output(0);

    dX->ResizeLike(X);
    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.ndim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.ndim() - num_reduce_dims_);

    const float* dYdata = dY.template data<float>();
    const float* Xdata = X.template data<float>();
    const float* Ydata = Y.template data<float>();

    float* dXdata = dX->template mutable_data<float>();
    Compute(rows, cols, dYdata, Xdata, Ydata, dXdata);
    return true;
  }

 protected:
  void Compute(
      int rows,
      int cols,
      const float* dYdata,
      const float* Xdata,
      const float* Ydata,
      float* dXdata);

  int num_reduce_dims_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCTION_FRONT_BACK_OPS_H_
