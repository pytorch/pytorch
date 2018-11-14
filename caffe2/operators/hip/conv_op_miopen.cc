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
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default miopen workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

class MIOPENConvOpBase : public ConvPoolOpBase<HIPContext> {
 public:
  MIOPENConvOpBase(const OperatorDef& operator_def, Workspace* ws)
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
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));

    if(group_ > 1) {
        mode_ = miopenGroupConv;
    } else{
        mode_ = miopenConvolution;
    }

    if (mode_ == miopenGroupConv) {
      OPERATOR_NEEDS_FEATURE(
          dilation_h() == 1 && dilation_w() == 1,
          "MIOpen convolution does not support dilation for groups > 1.");
    }
  }

  ~MIOPENConvOpBase() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bottom_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bias_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(weight_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_));
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_for_bias_));
    MIOPEN_ENFORCE(miopenDestroyConvolutionDescriptor(conv_desc_));
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
  size_t miopen_state_;
  const size_t miopen_ws_nbytes_limit_;
  bool exhaustive_search_;
  const float alpha_;
  const float beta_;
};

class MIOPENConvOp final : public MIOPENConvOpBase {
 public:
  MIOPENConvOp(const OperatorDef& operator_def, Workspace* ws)
      : MIOPENConvOpBase(operator_def, ws),
        requestAlgoCount_(
            OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
        returnedAlgoCount_(
            OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
        bestAlgoFound_(
            OperatorBase::GetSingleArgument<bool>("bestAlgoFound_", false)),
        fwdConvWs_(nullptr),
        fwdConvWsSize_(0),
        fwdAlgo_(miopenConvolutionFwdAlgoGEMM) {}

  ~MIOPENConvOp() {
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
};

class MIOPENConvGradientOp final : public MIOPENConvOpBase {
 public:
  MIOPENConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : MIOPENConvOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)),
        requestAlgoCount_(
            OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
        returnedAlgoCount_(
            OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
        bestDataAlgoFound_(
            OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)),
        bestWeightAlgoFound_(
            OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)),
        bwdWeiAlgo_(miopenConvolutionBwdWeightsAlgoGEMM),
        bwdDataAlgo_(miopenConvolutionBwdDataAlgoGEMM),
        bwdWeightWsSize_(0),
        bwdDataWsSize_(0),
        bwdWeightWs_(nullptr),
        bwdDataWs_(nullptr) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  ~MIOPENConvGradientOp() {
    if (bwdWeightWs_) {
      hipFree(bwdWeightWs_);
      bwdWeightWs_ = nullptr;
      bwdWeightWsSize_ = 0;
    }
    if (bwdDataWs_) {
      hipFree(bwdDataWs_);
      bwdDataWs_ = nullptr;
      bwdDataWsSize_ = 0;
    }
  }

  template <
      typename T_X,
      typename T_DY,
      typename T_W,
      typename T_B,
      typename MATH,
      typename T_DX,
      typename T_DW,
      typename T_DB>
  bool DoRunWithType();
  bool RunOnDevice() override;

 private:
  bool no_bias_;
  const int requestAlgoCount_;
  int returnedAlgoCount_;
  bool bestDataAlgoFound_;
  bool bestWeightAlgoFound_;
  miopenConvBwdWeightsAlgorithm_t bwdWeiAlgo_;
  miopenConvBwdDataAlgorithm_t bwdDataAlgo_;
  size_t bwdWeightWsSize_;
  size_t bwdDataWsSize_;
  char* bwdWeightWs_;
  char* bwdDataWs_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool MIOPENConvOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& Weight = Input(FILTER);
  auto* Y = Output(0);

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(
      Weight.ndim() == 4,
      "Conv op with MIOpen engine is supported only for 2D convolutions");

  const int M = Weight.dim32(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, M);

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
  CAFFE_ENFORCE_EQ(Weight.dim32(1), C / group_);

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  bool input_changed = (X.dims() != mio_input_dims_);
  bool weight_changed = (Weight.dims() != mio_weight_dims_);

  if (input_changed || weight_changed) {
    VLOG(1) << "Changing MIOpen descriptor configurations.";
    if (input_changed) {
      mio_input_dims_ = X.dims().vec();
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          bottom_desc_, miopenTypeWrapper<T_X>::type, N, C, H, W));
    }

    if (weight_changed) {
      mio_weight_dims_ = Weight.dims().vec();
      MIOPEN_ENFORCE(miopenInitConvolutionDescriptor(
          conv_desc_,
          mode_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w()));

      MIOPEN_ENFORCE(miopenSetConvolutionGroupCount(
          conv_desc_, group_));

      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          weight_desc_,
          miopenTypeWrapper<T_W>::type,
          M,
          C / group_,
          kernel_h(),
          kernel_w()));
    }

    MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(
        conv_desc_,
        bottom_desc_,
        weight_desc_,
        &N_out,
        &C_out,
        &H_out,
        &W_out));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        top_desc_, miopenTypeWrapper<T_X>::type, N_out, C_out, H_out, W_out));

    if (InputSize() == 3) {
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          bias_desc_, miopenTypeWrapper<T_B>::type, 1, M, 1, 1));
    }

    while (!bestAlgoFound_) {
      miopenConvAlgoPerf_t perf;

      MIOPEN_ENFORCE(miopenConvolutionForwardGetWorkSpaceSize(
          miopen_wrapper_.inline_miopen_handle(),
          weight_desc_,
          bottom_desc_,
          conv_desc_,
          top_desc_,
          &fwdConvWsSize_));
      if ((fwdConvWsSize_ > 0) && (fwdConvWs_ == nullptr)) {
          HIP_CHECK(hipMalloc(&fwdConvWs_, fwdConvWsSize_));
      }

      miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
          MIOPEN_ENFORCE(miopenFindConvolutionForwardAlgorithm(
            state->miopen_handle(),
            bottom_desc_,
            X.template data<T_X>(),
            weight_desc_,
            Weight.template data<T_W>(),
            conv_desc_,
            top_desc_,
            Y->template mutable_data<T_Y>(),
            requestAlgoCount_,
            &returnedAlgoCount_,
            &perf,
            fwdConvWs_,
            fwdConvWsSize_,
            false));
      });
      bestAlgoFound_ = true;
      fwdAlgo_ = perf.fwd_algo;
    }
  }

  miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
      MIOPEN_ENFORCE(miopenConvolutionForward(
        state->miopen_handle(),
        &alpha_,
        bottom_desc_,
        X.template data<T_X>(),
        weight_desc_,
        Weight.template data<T_W>(),
        conv_desc_,
        fwdAlgo_,
        &beta_,
        top_desc_,
        Y->template mutable_data<T_Y>(),
        fwdConvWs_,
        fwdConvWsSize_));
  });

  if (InputSize() == 3) {
      auto& bias = Input(BIAS);

      CAFFE_ENFORCE_EQ(bias.ndim(), 1);
      CAFFE_ENFORCE_EQ(bias.dim32(0), M);
      miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
        MIOPEN_ENFORCE(miopenConvolutionForwardBias(
          state->miopen_handle(),
          &alpha_,
          bias_desc_,
          bias.template data<T_B>(),
          &beta_,
          top_desc_,
          Y->template mutable_data<T_Y>()));
      });
  }
  return true;
}

// TODO : enable fp16 support.
bool MIOPENConvOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Math
        float>(); // Y
  } else {
    LOG(FATAL) << "Only float (32bit) is supported by "
               << "miopen convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}

template <
    typename T_X,
    typename T_DY,
    typename T_W,
    typename T_B,
    typename MATH,
    typename T_DX,
    typename T_DW,
    typename T_DB>
bool MIOPENConvGradientOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& Weight = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dW = Output(FILTER_GRAD);
  auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
  dX->ResizeLike(X);
  dW->ResizeLike(Weight);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(
      Weight.ndim() == 4,
      "ConvGradient op with MIOpen engine is supported only for 2D convolutions");

  const int M = Weight.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, D = 0, N_out = 0, C_out = 0, H_out = 0,
      W_out = 0, D_out = 0;

  N = X.dim32(0);
  C = X.dim32(1);
  H = X.dim32(2);
  W = X.ndim() > 3 ? X.dim32(3) : 1;
  D = X.ndim() > 4 ? X.dim32(4) : 1;

  N_out = dY.dim32(0);
  C_out = dY.dim32(1);
  H_out = dY.dim32(2);
  W_out = dY.ndim() > 3 ? dY.dim32(3) : 1;
  D_out = dY.ndim() > 4 ? dY.dim32(4) : 1;

  CAFFE_ENFORCE_EQ(Weight.dim32(1), C / group_);

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  bool doBwdDataComputation = (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2)));
  bool input_changed = (X.dims() != mio_input_dims_);
  bool weight_changed = (Weight.dims() != mio_weight_dims_);

  if (input_changed || weight_changed) {
    VLOG(1) << "Changing MIOpen descriptor configurations.";
    if (input_changed) {
      mio_input_dims_ = X.dims().vec();
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          bottom_desc_, miopenTypeWrapper<T_X>::type, N, C, H, W));
    }

    if (weight_changed) {
      mio_weight_dims_ = Weight.dims().vec();
      MIOPEN_ENFORCE(miopenInitConvolutionDescriptor(
          conv_desc_,
          mode_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w()));

      MIOPEN_ENFORCE(miopenSetConvolutionGroupCount(
          conv_desc_, group_));

      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          weight_desc_,
          miopenTypeWrapper<T_X>::type,
          M,
          C / group_,
          kernel_h(),
          kernel_w()));
    }

    MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(
        conv_desc_,
        bottom_desc_,
        weight_desc_,
        &N_out,
        &C_out,
        &H_out,
        &W_out));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        top_desc_, miopenTypeWrapper<T_X>::type, N_out, C_out, H_out, W_out));

    if (!no_bias_) {
        MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
            bias_desc_, miopenTypeWrapper<T_B>::type, 1, M, 1, 1));
    }

    while ((!bestDataAlgoFound_) && doBwdDataComputation) {
      miopenConvAlgoPerf_t perf;

      MIOPEN_ENFORCE(miopenConvolutionBackwardDataGetWorkSpaceSize(
          miopen_wrapper_.inline_miopen_handle(),
          top_desc_,
          weight_desc_,
          conv_desc_,
          bottom_desc_,
          &bwdDataWsSize_));
      if ((bwdDataWsSize_ > 0) && (bwdDataWs_ == nullptr)) {
          HIP_CHECK(hipMalloc(&bwdDataWs_, bwdDataWsSize_));
      }

      miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
        MIOPEN_ENFORCE(miopenFindConvolutionBackwardDataAlgorithm(
          state->miopen_handle(),
          top_desc_,
          dY.template data<T_DY>(),
          weight_desc_,
          Weight.template data<T_W>(),
          conv_desc_,
          bottom_desc_,
          dX->template mutable_data<T_DX>(),
          requestAlgoCount_,
          &returnedAlgoCount_,
          &perf,
          bwdDataWs_,
          bwdDataWsSize_,
          false));
      });

      bestDataAlgoFound_ = true;
      bwdDataAlgo_ = perf.bwd_data_algo;
  }

    while (!bestWeightAlgoFound_) {
        miopenConvAlgoPerf_t perf;

        MIOPEN_ENFORCE(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            miopen_wrapper_.inline_miopen_handle(),
            top_desc_,
            bottom_desc_,
            conv_desc_,
            weight_desc_,
            &bwdWeightWsSize_));
        if ((bwdWeightWsSize_ > 0) && (bwdWeightWs_ == nullptr)) {
          HIP_CHECK(hipMalloc(&bwdWeightWs_, bwdWeightWsSize_));
        }

        miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
          MIOPEN_ENFORCE(miopenFindConvolutionBackwardWeightsAlgorithm(
            state->miopen_handle(),
            top_desc_,
            dY.template data<T_DY>(),
            bottom_desc_,
            X.template data<T_X>(),
            conv_desc_,
            weight_desc_,
            dW->template mutable_data<T_DW>(),
            requestAlgoCount_,
            &returnedAlgoCount_,
            &perf,
            bwdWeightWs_,
            bwdWeightWsSize_,
            false));
        });
        bestWeightAlgoFound_ = true;
        bwdWeiAlgo_ = perf.bwd_weights_algo;
    }
  }
  if (doBwdDataComputation) {
      miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
        MIOPEN_ENFORCE(miopenConvolutionBackwardData(
          state->miopen_handle(),
          &alpha_,
          top_desc_,
          dY.template data<T_DY>(),
          weight_desc_,
          Weight.template data<T_W>(),
          conv_desc_,
          bwdDataAlgo_,
          &beta_,
          bottom_desc_,
          dX->template mutable_data<T_DX>(),
          bwdDataWs_,
          bwdDataWsSize_));
        });
  }

  miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
    MIOPEN_ENFORCE(miopenConvolutionBackwardWeights(
      state->miopen_handle(),
      &alpha_,
      top_desc_,
      dY.template data<T_DY>(),
      bottom_desc_,
      X.template data<T_X>(),
      conv_desc_,
      bwdWeiAlgo_,
      &beta_,
      weight_desc_,
      dW->template mutable_data<T_DW>(),
      bwdWeightWs_,
      bwdWeightWsSize_));
  });

  // Synchronize the work across groups.
  hipDeviceSynchronize();

  ////////////////////////////////////// BIAS ///////////////////////////
  if (!no_bias_) {
      auto* dbias = Output(BIAS_OR_INPUT_GRAD);
      dbias->Resize(M);
      miopen_wrapper_.with_miopen_state(miopen_state_, [&](MIOPENState* state) {
        MIOPEN_ENFORCE(miopenConvolutionBackwardBias(
          state->miopen_handle(),
          &alpha_,
          top_desc_,
          dY.template data<T_DY>(),
          &beta_,
          bias_desc_,
          dbias->template mutable_data<T_DB>()));
      });
  }

  return true;
}

bool MIOPENConvGradientOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, //  X
        float, // dY
        float, //  W
        float, //  b
        float, // Math
        float, // dX
        float, // dW
        float>(); // db
  } else {
    LOG(FATAL) << "Unsupported input types";
  }
  return true;
}

REGISTER_MIOPEN_OPERATOR(Conv, MIOPENConvOp);
REGISTER_MIOPEN_OPERATOR(ConvGradient, MIOPENConvGradientOp);
} // namespace caffe2
