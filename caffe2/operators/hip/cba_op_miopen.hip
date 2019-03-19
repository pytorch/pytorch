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

#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default miopen workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

class MIOPENConvBiasActivateOpBase : public ConvPoolOpBase<HIPContext> {
 public:
  MIOPENConvBiasActivateOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<HIPContext>(operator_def, ws),
        miopen_wrapper_(&context_),
        miopen_state_(
            OperatorBase::GetSingleArgument<size_t>("miopen_state", 0)),
        miopen_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>(
            "ws_nbytes_limit",
            kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES)),
        exhaustive_search_(
            OperatorBase::GetSingleArgument<bool>("exhaustive_search", false)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
        activ_alpha_(
            OperatorBase::GetSingleArgument<float>("activ_alpha", 1.0)),
        activ_beta_(OperatorBase::GetSingleArgument<float>("activ_beta", 0.0)),
        activ_gamma_(
            OperatorBase::GetSingleArgument<float>("activ_gamma", 0.0)),
        mode_(miopenConvolution),
        activ_mode_(miopenActivationRELU),
        fusePlanDesc(nullptr),
        fuseDirection(miopenVerticalFusion),
        conv_op(nullptr),
        bias_op(nullptr),
        activ_op(nullptr),
        fusionArgs(nullptr) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));
  }

  ~MIOPENConvBiasActivateOpBase() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bottom_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bias_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(weight_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenDestroyConvolutionDescriptor(conv_desc_));
    // fusion plan and args
    MIOPEN_ENFORCE(miopenDestroyFusionPlan(fusePlanDesc));
    MIOPEN_ENFORCE(miopenDestroyOperatorArgs(fusionArgs));
  }

 protected:
  vector<int64_t> mio_input_dims_;
  vector<int64_t> mio_weight_dims_;
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t bottom_desc_;
  miopenTensorDescriptor_t bias_desc_;
  miopenTensorDescriptor_t weight_desc_;
  miopenTensorDescriptor_t top_desc_;
  miopenTensorDescriptor_t top_desc_for_bias_;
  miopenConvolutionDescriptor_t conv_desc_;
  miopenConvolutionMode_t mode_;
  miopenActivationMode_t activ_mode_;
  miopenFusionPlanDescriptor_t fusePlanDesc;
  const miopenFusionDirection_t fuseDirection;
  miopenFusionOpDescriptor_t conv_op;
  miopenFusionOpDescriptor_t bias_op;
  miopenFusionOpDescriptor_t activ_op;
  miopenOperatorArgs_t fusionArgs;
  size_t miopen_state_;
  const size_t miopen_ws_nbytes_limit_;
  bool exhaustive_search_;
  const float alpha_;
  const float beta_;
  const float activ_alpha_;
  const float activ_beta_;
  const float activ_gamma_;
};

class MIOPENConvBiasActivateOp final : public MIOPENConvBiasActivateOpBase {
 public:
  MIOPENConvBiasActivateOp(const OperatorDef& operator_def, Workspace* ws)
      : MIOPENConvBiasActivateOpBase(operator_def, ws),
        requestAlgoCount_(
            OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
        returnedAlgoCount_(
            OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
        bestAlgoFound_(
            OperatorBase::GetSingleArgument<bool>("bestAlgoFound_", false)),
        fwdConvWs_(nullptr),
        fwdConvWsSize_(0),
        fwdAlgo_(miopenConvolutionFwdAlgoGEMM) {}

  ~MIOPENConvBiasActivateOp() {
    if (fwdConvWs_) {
      hipFree(fwdConvWs_);
      fwdConvWs_ = nullptr;
      fwdConvWsSize_ = 0;
    }
  }

  template <
      typename T_X,
      typename T_W,
      typename T_B,
      typename MATH,
      typename T_Y>
  bool DoRunWithType();
  bool RunOnDevice() override;

 private:
  const int requestAlgoCount_;
  int returnedAlgoCount_;
  bool bestAlgoFound_;
  char* fwdConvWs_;
  size_t fwdConvWsSize_;
  miopenConvFwdAlgorithm_t fwdAlgo_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};
////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool MIOPENConvBiasActivateOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& Weight = Input(FILTER);
  auto& bias = Input(BIAS);
  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(
      Weight.ndim() == 4,
      "Conv op with MIOpen engine is supported only for 2D convolutions");

  const int M = Weight.dim32(0);
  auto sizes = ConvPoolOpBase<HIPContext>::GetOutputSize(X, M);
  auto* Y = Output(0, sizes, at::dtype<T_Y>());

  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.ndim() > 3 ? X.dim32(3) : 1;
  int D = X.ndim() > 4 ? X.dim32(4) : 1;

  int N_out = Y->dim32(0);
  int C_out = Y->dim32(1);
  int H_out = Y->dim32(2);
  int W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
  int D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;

  if (N == 0) {
    Y->template mutable_data<T_Y>();
    return true;
  }

  // Creat a Fusion plan
  // TODO add cached fusion plan
  MIOPEN_ENFORCE(
      miopenCreateFusionPlan(&fusePlanDesc, fuseDirection, bottom_desc_));

  // Create and add operators
  MIOPEN_ENFORCE(miopenCreateOpConvForward(
      fusePlanDesc, &conv_op, conv_desc_, weight_desc_));
  MIOPEN_ENFORCE(miopenCreateOpBiasForward(fusePlanDesc, &bias_op, bias_desc_));
  MIOPEN_ENFORCE(
      miopenCreateOpActivationForward(fusePlanDesc, &activ_op, activ_mode_));

  // Compile the Fusion Plan
  miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
    MIOPEN_ENFORCE(
        miopenCompileFusionPlan(state->miopen_handle(), fusePlanDesc));
  });

  MIOPEN_ENFORCE(miopenSetOpArgsConvForward(
      fusionArgs, conv_op, &alpha_, &beta_, X.template data<T_X>()));
  MIOPEN_ENFORCE(miopenSetOpArgsActivForward(
      fusionArgs,
      activ_op,
      &alpha_,
      &beta_,
      activ_alpha_,
      activ_beta_,
      activ_gamma_));
  MIOPEN_ENFORCE(miopenSetOpArgsBiasForward(
      fusionArgs, bias_op, &alpha_, &beta_, bias.template data<T_B>()));

  // Execute a Fusion Plan
  miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
    MIOPEN_ENFORCE(miopenExecuteFusionPlan(
        state->miopen_handle(),
        fusePlanDesc,
        bottom_desc_,
        X.template data<T_X>(),
        top_desc_,
        Y->template data<T_Y>(),
        fusionArgs));
  });
  return true;
}

bool MIOPENConvBiasActivateOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Math
        float>(); // Y
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<
        at::Half, // X
        at::Half, // W
        at::Half, // B
        at::Half, // Math
        at::Half>(); // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and Half are supported by "
               << "miopen convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}

REGISTER_MIOPEN_OPERATOR(ConvBiasActivate, MIOPENConvBiasActivateOp);
} // namespace caffe2
