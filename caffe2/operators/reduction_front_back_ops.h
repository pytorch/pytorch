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

template <class Context, bool FIRSTDIMS, bool NORMALIZE>
class SumReduceDimsOp final : public Operator<Context> {
 public:
  SumReduceDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long, float, double>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& X = Input(0);
    auto* Y = Output(0);

    CAFFE_ENFORCE(
        num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.dims().size(),
        "For N-dim input tensor, support num_reduce_dims in range [0, N].");

    vector<TIndex> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index =
        FIRSTDIMS ? X.dims().size() : X.dims().size() - num_reduce_dims_;
    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(X.dims()[i]);
    }
    Y->Resize(output_shape);

    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.ndim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.ndim() - num_reduce_dims_);

    if (cols == 0 || rows == 0) {
      return true;
    }

    const T* in_data = X.template data<T>();
    T* out_data = Y->template mutable_data<T>();
    Compute(rows, cols, in_data, out_data);

    return true;
  }

 private:
  template <typename T>
  void Compute(int rows, int cols, const T* in_data, T* out_data);
  int num_reduce_dims_;
};

template <class Context, bool FIRSTDIMS, bool NORMALIZE>
class SumReduceDimsGradientOp final : public Operator<Context> {
 public:
  SumReduceDimsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long, float, double>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& dY = Input(0);
    auto& input_1 = Input(1);
    auto* dX = Output(0);

    // In previous diff we changed the semantic: Input(1) was changed from
    // the shape of the input to the data tensor. This made the backward
    // computation incompatible with old models. To fix this, we check
    // the dimension and type of Input(1).
    if (input_1.ndim() == 1 && input_1.template IsType<TIndex>()) {
      // Input(1) is the shape of the input
      shape_.CopyFrom(input_1);
      // Copy first dims
      vector<TIndex> output_shape(
          shape_.template data<TIndex>(),
          shape_.template data<TIndex>() + shape_.size());
      dX->Resize(output_shape);
    } else {
      // Input(1) is data tensor X
      dX->ResizeLike(input_1);
    }

    const int rows = FIRSTDIMS ? dX->size_to_dim(num_reduce_dims_)
                               : dX->size_to_dim(dX->ndim() - num_reduce_dims_);
    const int cols = FIRSTDIMS
        ? dX->size_from_dim(num_reduce_dims_)
        : dX->size_from_dim(dX->ndim() - num_reduce_dims_);

    const T* dYdata = dY.template data<T>();
    T* dXdata = dX->template mutable_data<T>();
    Compute<T>(rows, cols, dYdata, dXdata);
    return true;
  }

 private:
  template <typename T>
  void Compute(int rows, int cols, const T* dYdata, T* dXdata);
  int num_reduce_dims_;
  // scratch space used for former version of this reducer
  Tensor<CPUContext> shape_;
};

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

    CAFFE_ENFORCE(
        num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.dims().size(),
        "For N-dim input tensor, support num_reduce_dims in range [0, N].");

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
