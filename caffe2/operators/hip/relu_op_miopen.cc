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
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class MIOPENReluOp final : public Operator<HIPContext> {
 public:
  MIOPENReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
        power_(OperatorBase::GetSingleArgument<double>("power", 1.0)) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&data_desc_));
    MIOPEN_ENFORCE(miopenCreateActivationDescriptor(&activ_desc_));
    MIOPEN_ENFORCE(miopenSetActivationDescriptor(
        activ_desc_, miopenActivationRELU, alpha_, beta_, power_));
  }

  ~MIOPENReluOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(data_desc_));
    MIOPEN_ENFORCE(miopenDestroyActivationDescriptor(activ_desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);

    // Return if X is empty
    if (X.size() == 0) {
      Y->mutable_data<T>();
      return true;
    }

    // See if we need to reshape.
    if (X.dims() != miopen_input_dims_) {
      VLOG(1) << "Setting descriptors.";
      miopen_input_dims_ = X.dims();
      int C = 1, H = 1, W = 1;
      if (X.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = X.dim32(1);
        H = X.dim32(2);
        W = X.dim32(3);
      } else {
        // If X is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = X.size() / X.dim32(0);
      }
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          data_desc_, miopenTypeWrapper<T>::type, X.dim32(0), C, H, W));
    }
    MIOPEN_ENFORCE(miopenActivationForward(
        miopen_wrapper_.inline_miopen_handle(),
        activ_desc_,
        &alpha_,
        data_desc_,
        X.template data<T>(),
        &beta_,
        data_desc_,
        Y->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    // dispatch based on contents of tensor(s)
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    if (X.IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t data_desc_;
  miopenActivationDescriptor_t activ_desc_;
  vector<TIndex> miopen_input_dims_;
  const float alpha_;
  const float beta_;
  const double power_;
};

// Note: You can see that in MIOPENReluGradientOp, we abused the miopen
// interface by passing in the output tensor for both bottom and top. This is
// dependent on the assumption that the Relu gradient actually does not rely on
// the bottom data, or it treats input=0 the same way as input<0. This is of
// course not very safe, but we have been running in this way in Caffe for a
// while so it *might* be safe to assume so.
class MIOPENReluGradientOp final : public Operator<HIPContext> {
 public:
  MIOPENReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
        power_(OperatorBase::GetSingleArgument<double>("power", 1.0)) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&data_desc_));
    MIOPEN_ENFORCE(miopenCreateActivationDescriptor(&activ_desc_));
    MIOPEN_ENFORCE(miopenSetActivationDescriptor(
        activ_desc_, miopenActivationRELU, alpha_, beta_, power_));
  }

  ~MIOPENReluGradientOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(data_desc_));
    MIOPEN_ENFORCE(miopenDestroyActivationDescriptor(activ_desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);

    // Return if Y is empty
    if (Y.size() == 0) {
      dX->mutable_data<T>();
      return true;
    }

    // See if we need to reshape.
    if (Y.dims() != miopen_input_dims_) {
      VLOG(1) << "Setting descriptors.";
      miopen_input_dims_ = Y.dims();
      int C = 1, H = 1, W = 1;
      if (Y.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = Y.dim32(1);
        H = Y.dim32(2);
        W = Y.dim32(3);
      } else {
        // If Y is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = Y.size() / Y.dim32(0);
      }
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          data_desc_, miopenTypeWrapper<T>::type, Y.dim32(0), C, H, W));
    }
    MIOPEN_ENFORCE(miopenActivationBackward(
        miopen_wrapper_.inline_miopen_handle(),
        activ_desc_,
        &alpha_,
        data_desc_,
        Y.template data<T>(),
        data_desc_,
        dY.template data<T>(),
        data_desc_,
        Y.template data<T>(),
        &beta_,
        data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    const auto& Y = Input(0);
    auto* dX = Output(0);
    dX->ResizeLike(Y);
    if (Y.IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t data_desc_;
  miopenActivationDescriptor_t activ_desc_;
  vector<TIndex> miopen_input_dims_;
  const float alpha_;
  const float beta_;
  const double power_;
  // Input: Y, dY; Output: dX
};

namespace {
REGISTER_MIOPEN_OPERATOR(Relu, MIOPENReluOp);
REGISTER_MIOPEN_OPERATOR(ReluGradient, MIOPENReluGradientOp);
} // namespace
} // namespace caffe2
