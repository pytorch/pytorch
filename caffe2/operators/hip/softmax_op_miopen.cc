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

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/softmax_op.h"

namespace caffe2 {
class MIOpenSoftmaxOp final : public Operator<HIPContext> {
 public:
  explicit MIOpenSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<HIPContext>(def, ws),
        miopen_wrapper_(&context_),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&desc_));
  }

  ~MIOpenSoftmaxOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& X = Input(0);
    auto* Y = Output(0);
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int N = X.size_to_dim(canonical_axis);
    const int D = X.size_from_dim(canonical_axis);

    Y->ResizeLike(X);
    if (dims_ != X.dims()) {
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          desc_, miopenTypeWrapper<T>::type, N, D, 1, 1));
      dims_ = X.dims();
    }
    MIOPEN_ENFORCE(miopenSoftmaxForward(
        miopen_wrapper_.inline_miopen_handle(),
        &alpha_,
        desc_,
        X.template data<T>(),
        &beta_,
        desc_,
        Y->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
  }

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t desc_;
  vector<TIndex> dims_;
  const int axis_;
  const float alpha_;
  const float beta_;
};

class MIOpenSoftmaxGradientOp final : public Operator<HIPContext> {
 public:
  explicit MIOpenSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<HIPContext>(def, ws),
        miopen_wrapper_(&context_),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&desc_));
  }

  ~MIOpenSoftmaxGradientOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& Y = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    const auto canonical_axis = Y.canonical_axis_index(axis_);
    const int N = Y.size_to_dim(canonical_axis);
    const int D = Y.size_from_dim(canonical_axis);

    CHECK_EQ(Y.dims(), dY.dims());
    dX->ResizeLike(Y);
    if (dims_ != Y.dims()) {
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          desc_, miopenTypeWrapper<T>::type, N, D, 1, 1));
      dims_ = Y.dims();
    }
    MIOPEN_ENFORCE(miopenSoftmaxBackward(
        miopen_wrapper_.inline_miopen_handle(),
        &alpha_,
        desc_,
        Y.template data<T>(),
        desc_,
        dY.template data<T>(),
        &beta_,
        desc_,
        dX->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
  }

 protected:
  MIOPENWrapper miopen_wrapper_;
  const int axis_;
  const float alpha_;
  const float beta_;
  miopenTensorDescriptor_t desc_;
  vector<TIndex> dims_;
};

namespace {
REGISTER_MIOPEN_OPERATOR(Softmax, MIOpenSoftmaxOp);
REGISTER_MIOPEN_OPERATOR(SoftmaxGradient, MIOpenSoftmaxGradientOp);
} // namespace

} // namespace caffe2
