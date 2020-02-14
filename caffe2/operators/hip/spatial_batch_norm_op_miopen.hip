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

#include <cfloat>
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/operators/spatial_batch_norm_op.h"
#include "caffe2/operators/hip/spatial_batch_norm_op_impl.cuh"
#include "caffe2/utils/math.h"

const double MIOPEN_BN_MIN_EPSILON = 1e-6;

namespace caffe2 {

class MIOpenSpatialBNOp final : public SpatialBNOp<HIPContext> {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);
  MIOpenSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
        mode_(miopenBNSpatial) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&data_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon_ <= MIOPEN_BN_MIN_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "MIOPEN_BN_MIN_EPSILON. Setting it to "
                 << "MIOPEN_BN_MIN_EPSILON instead.";
    }
    epsilon_ = std::max(epsilon_, MIOPEN_BN_MIN_EPSILON);
  }

  ~MIOpenSpatialBNOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(data_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bn_param_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();
  bool RunOnDevice() override;

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t data_desc_;
  miopenTensorDescriptor_t bn_param_desc_;
  vector<int64_t> miopen_input_dims_;
  float alpha_;
  float beta_;
  miopenBatchNormMode_t mode_;
};

class MIOpenSpatialBNGradientOp final : public SpatialBNGradientOp<HIPContext> {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);
  MIOpenSpatialBNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNGradientOp<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
        mode_(miopenBNSpatial) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&data_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon_ <= MIOPEN_BN_MIN_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "MIOPEN_BN_MIN_EPSILON. Setting it to "
                 << "MIOPEN_BN_MIN_EPSILON instead.";
    }
    epsilon_ = std::max(epsilon_, MIOPEN_BN_MIN_EPSILON);
  }

  ~MIOpenSpatialBNGradientOp() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(data_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bn_param_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();

  bool RunOnDevice() override;

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t data_desc_;
  miopenTensorDescriptor_t bn_param_desc_;
  vector<int64_t> miopen_input_dims_;
  float alpha_;
  float beta_;
  miopenBatchNormMode_t mode_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename M>
bool MIOpenSpatialBNOp::DoRunWithType() {
  // QoL
  typedef typename miopenTypeWrapper<T>::BNParamType BNParamType;

  auto& X = Input(INPUT);
  auto& scale = Input(SCALE);
  auto& bias = Input(BIAS);
  auto* Y = Output(OUTPUT);

  // Only 2D and 3D BatchNorm are supported in MIopen for now
  CAFFE_ENFORCE(
        X.ndim() == 4 || X.ndim() == 5,
        "Currently only 2D and 3D batchnorms are supported with MIOpen.");
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.ndim() > 3 ? X.dim32(3) : 1;
  const int D = X.ndim() > 4 ? X.dim32(4) : 1;
  CAFFE_ENFORCE_EQ(scale.ndim(), 1);
  CAFFE_ENFORCE_EQ(bias.ndim(), 1);
  CAFFE_ENFORCE_EQ(scale.dim32(0), C);
  CAFFE_ENFORCE_EQ(bias.dim32(0), C);

  Y->ResizeLike(X);
  T* Y_data = Y->template mutable_data<T>();

  // See if we need to reshape.
  if (N > 0 && X.sizes() != miopen_input_dims_) {
    VLOG(1) << "Setting descriptors.";
    miopen_input_dims_ = X.sizes().vec();
    vector<int> input_dims(begin(miopen_input_dims_), end(miopen_input_dims_));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        data_desc_, miopenTypeWrapper<T>::type, input_dims.size(), input_dims.data(), nullptr));

    MIOPEN_ENFORCE(
        miopenDeriveBNTensorDescriptor(bn_param_desc_, data_desc_, mode_));
  }

  // Now, depending on whether we are running test or not, we have two paths.
  if (is_test_) {
    // Run inference mode.
    auto& est_mean = Input(EST_MEAN);
    auto& est_var = Input(EST_VAR);
    CAFFE_ENFORCE_EQ(est_mean.ndim(), 1);
    CAFFE_ENFORCE_EQ(est_var.ndim(), 1);
    CAFFE_ENFORCE_EQ(est_mean.dim32(0), C);
    CAFFE_ENFORCE_EQ(est_var.dim32(0), C);

    if (N == 0) {
      return true;
    }
    MIOPEN_ENFORCE(miopenBatchNormalizationForwardInference(
        miopen_wrapper_.inline_miopen_handle(),
        // Note: PERSISTENT not implemented for inference
        mode_,
        &alpha_,
        &beta_,
        data_desc_,
        X.template data<T>(),
        data_desc_,
        Y_data,
        bn_param_desc_,
        const_cast<float*>(scale.template data<BNParamType>()),
        const_cast<float*>(bias.template data<BNParamType>()),
        const_cast<float*>(est_mean.template data<BNParamType>()),
        const_cast<float*>(est_var.template data<BNParamType>()),
        epsilon_));
  } else {
    // Run training mode.
    // obtain running mean and running inv var, and see if we need to
    // initialize them.
    auto* running_mean = Output(RUNNING_MEAN);
    auto* running_var = Output(RUNNING_VAR);
    double this_factor = 1. - momentum_;
    BNParamType* running_mean_data = nullptr;
    BNParamType* running_var_data = nullptr;
    if (!running_mean->size()) {
      // If the input mean and var are not initialized yet, this is the first
      // run and we will initialize the storage.
      VLOG(1) << "Initializing running mean and var.";
      // Need to do initialization
      running_mean->Resize(C);
      running_var->Resize(C);
      running_mean_data = running_mean->template mutable_data<BNParamType>();
      running_var_data = running_var->template mutable_data<BNParamType>();
      // In principle, setting this_momentum to 1 will wipe existing data.
      // This has a caveat that if miopen does not deal with 0*NaN cases we
      // will be having an issue. Thus we choose a safe path by explicitly
      // setting zero.
      math::Set<BNParamType, HIPContext>(C, 0, running_mean_data, &context_);
      math::Set<BNParamType, HIPContext>(C, 0, running_var_data, &context_);
    } else {
      // Does not need to do initialization.
      CAFFE_ENFORCE_EQ(running_mean->ndim(), 1);
      CAFFE_ENFORCE_EQ(running_var->ndim(), 1);
      CAFFE_ENFORCE_EQ(running_mean->dim32(0), C);
      CAFFE_ENFORCE_EQ(running_var->dim32(0), C);
      running_mean_data = running_mean->template mutable_data<BNParamType>();
      running_var_data = running_var->template mutable_data<BNParamType>();
    }
    // Save the mean and inv var results.
    auto* save_mean = Output(SAVED_MEAN);
    auto* save_var = Output(SAVED_INV_STD);
    save_mean->Resize(C);
    save_var->Resize(C);
    void* save_mean_data = save_mean->template mutable_data<BNParamType>();
    void* save_var_data = save_var->template mutable_data<BNParamType>();

    if (N == 0) {
      // set empty batch's mean / var output to zeros
      math::Set<BNParamType, HIPContext>(
          C, 0, (BNParamType*)save_mean_data, &context_);
      math::Set<BNParamType, HIPContext>(
          C, 0, (BNParamType*)save_var_data, &context_);
      return true;
    }
    MIOPEN_ENFORCE(miopenBatchNormalizationForwardTraining(
        miopen_wrapper_.inline_miopen_handle(),
        mode_,
        &alpha_,
        &beta_,
        data_desc_,
        X.template data<T>(),
        data_desc_,
        Y_data,
        bn_param_desc_,
        const_cast<float*>(scale.template data<BNParamType>()),
        const_cast<float*>(bias.template data<BNParamType>()),
        this_factor,
        const_cast<float*>(running_mean_data),
        const_cast<float*>(running_var_data),
        epsilon_,
        save_mean_data,
        save_var_data));
  }
  return true;
}
bool MIOpenSpatialBNOp::RunOnDevice() {
  // Fall back to HIP for multi batch spatial BNorm and NHWC layout
  if (num_batches_ > 1 || order_ == StorageOrder::NHWC) {
    return SpatialBNOp<HIPContext>::RunOnDevice();
  }
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (Input(0).IsType<at::Half>()){
    return DoRunWithType<at::Half, float>();
  } else {
    LOG(FATAL) << "Unsupported input types";
  }
  return true;
}

template <typename T, typename M>
bool MIOpenSpatialBNGradientOp::DoRunWithType() {
  typedef typename miopenTypeWrapper<T>::BNParamType BNParamType;

  auto& X = Input(INPUT);
  auto& scale = Input(SCALE);
  auto& dY = Input(OUTPUT_GRAD);

  CAFFE_ENFORCE_GE(X.ndim(), 3);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.ndim() > 3 ? X.dim32(3) : 1;
  const int D = X.ndim() > 4 ? X.dim32(4) : 1;
  CAFFE_ENFORCE_EQ(scale.ndim(), 1);
  CAFFE_ENFORCE_EQ(scale.dim32(0), C);
  // See if we need to reshape.
  if (N > 0 && X.sizes() != miopen_input_dims_) {
    miopen_input_dims_ = X.sizes().vec();
    vector<int> input_dims(begin(miopen_input_dims_), end(miopen_input_dims_));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        data_desc_, miopenTypeWrapper<T>::type, input_dims.size(), input_dims.data(), nullptr));

    MIOPEN_ENFORCE(
        miopenDeriveBNTensorDescriptor(bn_param_desc_, data_desc_, mode_));
  }

  auto* dX = Output(INPUT_GRAD);
  dX->ResizeLike(X);
  T* dX_data = dX->template mutable_data<T>();
  auto* dScale = Output(SCALE_GRAD);
  auto* dBias = Output(BIAS_GRAD);
  dX->ResizeLike(X);
  dScale->ResizeLike(scale);
  dBias->ResizeLike(scale);
  auto* dScale_data = dScale->template mutable_data<BNParamType>();
  auto* dBias_data = dBias->template mutable_data<BNParamType>();

  const auto& saved_mean = Input(SAVED_MEAN);
  const auto& saved_var = Input(SAVED_INV_STD);
  const void* saved_mean_data = saved_mean.template data<BNParamType>();
  const void* saved_var_data = saved_var.template data<BNParamType>();

  if (N == 0) {
    // set gradients to zeros
    math::Set<BNParamType, HIPContext>(C, 0, dScale_data, &context_);
    math::Set<BNParamType, HIPContext>(C, 0, dBias_data, &context_);
    return true;
  }

  MIOPEN_ENFORCE(miopenBatchNormalizationBackward(
      miopen_wrapper_.inline_miopen_handle(),
      mode_,
      &alpha_,
      &beta_,
      &alpha_,
      &beta_,
      data_desc_,
      X.template data<T>(),
      data_desc_,
      dY.template data<T>(),
      data_desc_,
      dX_data,
      bn_param_desc_,
      scale.template data<BNParamType>(),
      dScale_data,
      dBias_data,
      epsilon_,
      saved_mean_data,
      saved_var_data));
  return true;
}
bool MIOpenSpatialBNGradientOp::RunOnDevice() {
  // Fall back to HIP for multi batch spatial BNorm and NHWC layout
  if (num_batches_ > 1 || order_ == StorageOrder::NHWC) {
    return SpatialBNGradientOp<HIPContext>::RunOnDevice();
  }
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<at::Half, float>();
  } else {
    LOG(FATAL) << "Unsupported input types";
  }
  return true;
}

REGISTER_MIOPEN_OPERATOR(SpatialBN, MIOpenSpatialBNOp);
REGISTER_MIOPEN_OPERATOR(SpatialBNGradient, MIOpenSpatialBNGradientOp);
} // namespace caffe2
