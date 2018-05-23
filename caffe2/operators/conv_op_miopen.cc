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

#include "caffe2/operators/conv_op.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/core/miopen_wrapper.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default miopen workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

class MIOPENConvOpBase : public ConvPoolOpBase<HIPContext>
{
    public:
    MIOPENConvOpBase(const OperatorDef& operator_def, Workspace* ws)
        : ConvPoolOpBase<HIPContext>(operator_def, ws),
          miopen_wrapper_(&context_),
          miopen_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>(
              "ws_nbytes_limit", kCONV_MIOPEN_WORKSPACE_LIMIT_BYTES)),
          alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)),
          beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)),
          exhaustive_search_(OperatorBase::GetSingleArgument<bool>("exhaustive_search", false))
    {
        MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
        MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
        MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
        MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
        MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
        MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));

        if((operator_def.type().substr(0, 6) == "Conv") ||
           (operator_def.type().substr(0, 14) == "ConvGradient"))
        {
            mode_ = miopenConvolution;
        }
        else if((operator_def.type().substr(0, 7) == "Trans") ||
                (operator_def.type().substr(0, 15) == "TransGradient"))
        {
            mode_ = miopenTranspose;
        }
        else
        {
            LOG(FATAL) << "Unsupported convolution method: " << operator_def.type();
        }

        MIOPEN_ENFORCE(miopenInitConvolutionDescriptor(conv_desc_,
                                                       mode_,
                                                       pad_t(),
                                                       pad_l(),
                                                       stride_h(),
                                                       stride_w(),
                                                       dilation_h(),
                                                       dilation_w()));
    }

    ~MIOPENConvOpBase()
    {
        MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bottom_desc_));
        MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(bias_desc_));
        MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(weight_desc_));
        MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_));
        MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(top_desc_for_bias_));
        MIOPEN_ENFORCE(miopenDestroyConvolutionDescriptor(conv_desc_));
    }

    protected:
    MIOPENWrapper miopen_wrapper_;
    miopenTensorDescriptor_t bottom_desc_;
    miopenTensorDescriptor_t bias_desc_;
    miopenTensorDescriptor_t weight_desc_;
    miopenTensorDescriptor_t top_desc_;
    miopenTensorDescriptor_t top_desc_for_bias_;
    miopenConvolutionDescriptor_t conv_desc_;
    miopenConvolutionMode_t mode_;
    const size_t miopen_ws_nbytes_limit_;
    bool exhaustive_search_;
    const float alpha_;
    const float beta_;
};

class MIOPENConvOp final : public MIOPENConvOpBase
{
    public:
    MIOPENConvOp(const OperatorDef& operator_def, Workspace* ws)
        : MIOPENConvOpBase(operator_def, ws),
          requestAlgoCount_(OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
          returnedAlgoCount_(OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
          bestAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound_", false)),
          fwdConvWs(nullptr),
          fwdConvWsSize_(0),
          fwd_algo_(miopenConvolutionFwdAlgoGEMM)
    {
    }

    ~MIOPENConvOp()
    {
        if(fwdConvWs)
        {
            hipFree(fwdConvWs);
            fwdConvWs      = nullptr;
            fwdConvWsSize_ = 0;
        }
    }

    template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
    bool DoRunWithType();
    bool RunOnDevice() override;

    private:
    const int requestAlgoCount_;
    int returnedAlgoCount_;
    bool bestAlgoFound_;
    char* fwdConvWs;
    size_t fwdConvWsSize_;
    miopenConvFwdAlgorithm_t fwd_algo_;
    // Input: X, W, b
    // Output: Y
    INPUT_TAGS(INPUT, FILTER, BIAS);
};

class MIOPENConvGradientOp final : public MIOPENConvOpBase
{
    public:
    MIOPENConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
        : MIOPENConvOpBase(operator_def, ws),
          no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)),
          requestAlgoCount_(OperatorBase::GetSingleArgument<int>("requestAlgoCount_", 1)),
          returnedAlgoCount_(OperatorBase::GetSingleArgument<int>("returnedAlgoCount_", 1)),
          bestDataAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)),
          bestWeightAlgoFound_(OperatorBase::GetSingleArgument<bool>("bestAlgoFound", false)),
          bwdWeightWs(nullptr),
          bwdWeightWsSize_(0),
          bwdDataWs(nullptr),
          bwdDataWsSize_(0),
          bwd_wei_algo_(miopenConvolutionBwdWeightsAlgoGEMM),
          bwd_data_algo_(miopenConvolutionBwdDataAlgoGEMM)
    {
        CAFFE_ENFORCE(!(no_bias_ && OutputSize() == 3),
                      "If bias is not present, you should not have 3 grad output.");
    }

    ~MIOPENConvGradientOp()
    {
        if(bwdWeightWs)
        {
            hipFree(bwdWeightWs);
            bwdWeightWs      = nullptr;
            bwdWeightWsSize_ = 0;
        }
        if(bwdDataWs)
        {
            hipFree(bwdDataWs);
            bwdDataWs      = nullptr;
            bwdDataWsSize_ = 0;
        }
    }

    template <typename T_X,
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
    miopenConvBwdWeightsAlgorithm_t bwd_wei_algo_;
    miopenConvBwdDataAlgorithm_t bwd_data_algo_;
    size_t bwdWeightWsSize_;
    size_t bwdDataWsSize_;
    char* bwdWeightWs;
    char* bwdDataWs;
    // input: X, W, dY
    // output: dW, db, and optionally dX
    INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
    OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool MIOPENConvOp::DoRunWithType()
{
    auto& X      = Input(INPUT);
    auto& Weight = Input(FILTER);
    auto* Y      = Output(0);

    // Figure out the output shape
    CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
    CAFFE_ENFORCE(Weight.ndim() == 4,
                  "Conv/Trans op with MIOpen engine is supported only for 2D convolutions");

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

    CAFFE_ENFORCE(C % group_ == 0,
                  "If you set group, the number of input channels should be divisible "
                  "by group.");
    CAFFE_ENFORCE(M % group_ == 0,
                  "If you set group, the number of output channels should be divisible "
                  "by group.");

    int group_offset_filter = Weight.size() / group_;

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(weight_desc_,
                                               miopenTypeWrapper<T_W>::type,
                                               M / group_,
                                               C / group_,
                                               kernel_h(),
                                               kernel_w()));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        bottom_desc_, miopenTypeWrapper<T_X>::type, N, C / group_, H, W));

    MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(
        conv_desc_, bottom_desc_, weight_desc_, &N_out, &C_out, &H_out, &W_out));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        top_desc_, miopenTypeWrapper<T_X>::type, N_out, C_out, H_out, W_out));

    if(InputSize() == 3)
    {
        MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
            bias_desc_, miopenTypeWrapper<T_B>::type, 1, Y->dim32(1), 1, 1));
        MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
            top_desc_for_bias_, miopenTypeWrapper<T_X>::type, N_out, Y->dim32(1), H_out, W_out));
    }

    MIOPEN_ENFORCE(miopenConvolutionForwardGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                            weight_desc_,
                                                            bottom_desc_,
                                                            conv_desc_,
                                                            top_desc_,
                                                            &fwdConvWsSize_));

    int group_offset_X = C / group_ * H * W * D;
    int group_offset_Y = M / group_ * H_out * W_out * D_out;

    fwdConvWsSize_ = (group_ > 1) ? miopen_ws_nbytes_limit_ : fwdConvWsSize_;

    if((fwdConvWsSize_ > 0) && (fwdConvWs == nullptr))
    {
        HIP_CHECK(hipMalloc(&fwdConvWs, fwdConvWsSize_));
    }

    while(!bestAlgoFound_)
    {
        miopenConvAlgoPerf_t perf;
        MIOPEN_ENFORCE(miopenFindConvolutionForwardAlgorithm(miopen_wrapper_.inline_miopen_handle(),
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
                                                             fwdConvWs,
                                                             fwdConvWsSize_,
                                                             false));
        bestAlgoFound_ = true;
        fwd_algo_      = perf.fwd_algo;
    }

    for(int i = 0; i < group_; i++)
    {
        MIOPEN_ENFORCE(
            miopenConvolutionForward(miopen_wrapper_.inline_miopen_handle(),
                                     &alpha_,
                                     bottom_desc_,
                                     X.template data<T_X>() + i * group_offset_X,
                                     weight_desc_,
                                     Weight.template data<T_W>() + i * group_offset_filter,
                                     conv_desc_,
                                     fwd_algo_,
                                     &beta_,
                                     top_desc_,
                                     Y->template mutable_data<T_Y>() + i * group_offset_Y,
                                     fwdConvWs,
                                     fwdConvWsSize_));
    }

    hipDeviceSynchronize();

    // BIAS
    if(InputSize() == 3)
    {
        auto& bias = Input(BIAS);

        CAFFE_ENFORCE_EQ(bias.ndim(), 1);
        CAFFE_ENFORCE_EQ(bias.dim32(0), M);
        MIOPEN_ENFORCE(miopenConvolutionForwardBias(miopen_wrapper_.inline_miopen_handle(),
                                                    &alpha_,
                                                    bias_desc_,
                                                    bias.template data<T_B>(),
                                                    &beta_,
                                                    top_desc_for_bias_,
                                                    Y->template mutable_data<T_Y>()));
    }

    hipDeviceSynchronize();
    return true;
}
// TODO : enable fp16 support.
bool MIOPENConvOp::RunOnDevice()
{
    if(Input(0).IsType<float>())
    {
        return DoRunWithType<float,    // X
                             float,    // W
                             float,    // B
                             float,    // Math
                             float>(); // Y
    }
    else
    {
        LOG(FATAL) << "Only float (32bit) is supported by "
                   << "miopen convolution, but input " << debug_def().input(0) << " has ["
                   << Input(0).meta().name() << "]";
    }
    return true;
}

template <typename T_X,
          typename T_DY,
          typename T_W,
          typename T_B,
          typename MATH,
          typename T_DX,
          typename T_DW,
          typename T_DB>
bool MIOPENConvGradientOp::DoRunWithType()
{
    auto& X      = Input(INPUT);
    auto& Weight = Input(FILTER);
    auto& dY     = Input(OUTPUT_GRAD);
    auto* dW     = Output(FILTER_GRAD);
    auto* dX     = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    dW->ResizeLike(Weight);

    CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
    CAFFE_ENFORCE(
        Weight.ndim() == 4,
        "ConvGradient/TransGradient op with MIOpen engine is supported only for 2D convolutions");

    const int M = Weight.dim32(0);
    int N = 0, C = 0, H = 0, W = 0, D = 0, N_out = 0, C_out = 0, H_out = 0, W_out = 0, D_out = 0;

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

    CAFFE_ENFORCE(C % group_ == 0,
                  "If you set group, the number of input channels should be divisible "
                  "by group.");
    CAFFE_ENFORCE(M % group_ == 0,
                  "If you set group, the number of output channels should be divisible "
                  "by group.");

    int group_offset_filter = Weight.size() / group_;

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(weight_desc_,
                                               miopenTypeWrapper<T_X>::type,
                                               M / group_,
                                               C / group_,
                                               kernel_h(),
                                               kernel_w()));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        bottom_desc_, miopenTypeWrapper<T_X>::type, N, C / group_, H, W));

    MIOPEN_ENFORCE(miopenGetConvolutionForwardOutputDim(
        conv_desc_, bottom_desc_, weight_desc_, &N_out, &C_out, &H_out, &W_out));

    MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
        top_desc_, miopenTypeWrapper<T_X>::type, N_out, C_out, H_out, W_out));

    if(!no_bias_)
    {
        MIOPEN_ENFORCE(
            miopenSet4dTensorDescriptor(bias_desc_, miopenTypeWrapper<T_B>::type, 1, M, 1, 1));
        MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
            top_desc_for_bias_, miopenTypeWrapper<T_X>::type, N_out, M, H_out, W_out));
    }

    MIOPEN_ENFORCE(
        miopenConvolutionBackwardDataGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                      top_desc_,
                                                      weight_desc_,
                                                      conv_desc_,
                                                      bottom_desc_,
                                                      &bwdDataWsSize_));

    int group_offset_X = C / group_ * H * W * D;
    int group_offset_Y = M / group_ * H_out * W_out * D_out;

    bwdDataWsSize_ = (group_ > 1) ? miopen_ws_nbytes_limit_ : bwdDataWsSize_;
    if((bwdDataWsSize_ > 0) && (bwdDataWs == nullptr))
    {
        HIP_CHECK(hipMalloc(&bwdDataWs, bwdDataWsSize_));
    }

    MIOPEN_ENFORCE(
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopen_wrapper_.inline_miopen_handle(),
                                                         top_desc_,
                                                         bottom_desc_,
                                                         conv_desc_,
                                                         weight_desc_,
                                                         &bwdWeightWsSize_));

    bwdWeightWsSize_ = (group_ > 1) ? miopen_ws_nbytes_limit_ : bwdWeightWsSize_;
    if((bwdWeightWsSize_ > 0) && (bwdWeightWs == nullptr))
    {
        HIP_CHECK(hipMalloc(&bwdWeightWs, bwdWeightWsSize_));
    }
    //////////// BWD DATA ////////////////////////////////////////

    for(int i = 0; i < group_; i++)
    {
        while(!bestDataAlgoFound_)
        {
            miopenConvAlgoPerf_t perf;
            MIOPEN_ENFORCE(miopenFindConvolutionBackwardDataAlgorithm(
                miopen_wrapper_.inline_miopen_handle(),
                top_desc_,
                dY.template data<T_DY>() + i * group_offset_Y,
                weight_desc_,
                Weight.template data<T_W>() + i * group_offset_filter,
                conv_desc_,
                bottom_desc_,
                dX->template mutable_data<T_DX>() + i * group_offset_X,
                requestAlgoCount_,
                &returnedAlgoCount_,
                &perf,
                bwdDataWs,
                bwdDataWsSize_,
                false));

            bestDataAlgoFound_ = true;
            bwd_data_algo_     = perf.bwd_data_algo;
        }

        MIOPEN_ENFORCE(
            miopenConvolutionBackwardData(miopen_wrapper_.inline_miopen_handle(),
                                          &alpha_,
                                          top_desc_,
                                          dY.template data<T_DY>() + i * group_offset_Y,
                                          weight_desc_,
                                          Weight.template data<T_W>() + i * group_offset_filter,
                                          conv_desc_,
                                          bwd_data_algo_,
                                          &beta_,
                                          bottom_desc_,
                                          dX->template mutable_data<T_DX>() + i * group_offset_X,
                                          bwdDataWs,
                                          bwdDataWsSize_));
        //////////////////////////////   BWD WEIGHT //////////////////////

        while(!bestWeightAlgoFound_)
        {
            miopenConvAlgoPerf_t perf;
            MIOPEN_ENFORCE(miopenFindConvolutionBackwardWeightsAlgorithm(
                miopen_wrapper_.inline_miopen_handle(),
                top_desc_,
                dY.template data<T_DY>() + i * group_offset_Y,
                bottom_desc_,
                X.template data<T_X>() + i * group_offset_X,
                conv_desc_,
                weight_desc_,
                dW->template mutable_data<T_DW>() + i * group_offset_filter,
                requestAlgoCount_,
                &returnedAlgoCount_,
                &perf,
                bwdWeightWs,
                bwdWeightWsSize_,
                false));
            bestWeightAlgoFound_ = true;
            bwd_wei_algo_        = perf.bwd_weights_algo;
        }
        MIOPEN_ENFORCE(miopenConvolutionBackwardWeights(
            miopen_wrapper_.inline_miopen_handle(),
            &alpha_,
            top_desc_,
            dY.template data<T_DY>() + i * group_offset_Y,
            bottom_desc_,
            X.template data<T_X>() + i * group_offset_X,
            conv_desc_,
            bwd_wei_algo_,
            &beta_,
            weight_desc_,
            dW->template mutable_data<T_DW>() + i * group_offset_filter,
            bwdWeightWs,
            bwdWeightWsSize_));
    }
    // Synchronize the work across groups.
    hipDeviceSynchronize();

    ////////////////////////////////////// BIAS ///////////////////////////
    if(!no_bias_)
    {
        auto* dbias = Output(BIAS_OR_INPUT_GRAD);
        dbias->Resize(M);
        MIOPEN_ENFORCE(miopenConvolutionBackwardBias(miopen_wrapper_.inline_miopen_handle(),
                                                     &alpha_,
                                                     top_desc_for_bias_,
                                                     dY.template data<T_DY>(),
                                                     &beta_,
                                                     bias_desc_,
                                                     dbias->template mutable_data<T_DB>()));
    }
    return true;
}

bool MIOPENConvGradientOp::RunOnDevice()
{
    if(Input(0).IsType<float>())
    {
        return DoRunWithType<float,    //  X
                             float,    // dY
                             float,    //  W
                             float,    //  b
                             float,    // Math
                             float,    // dX
                             float,    // dW
                             float>(); // db
    }
    else
    {
        LOG(FATAL) << "Unsupported input types";
    }
    return true;
}

REGISTER_MIOPEN_OPERATOR(Conv, MIOPENConvOp);
REGISTER_MIOPEN_OPERATOR(ConvGradient, MIOPENConvGradientOp);
REGISTER_MIOPEN_OPERATOR(Trans, MIOPENConvOp);
REGISTER_MIOPEN_OPERATOR(TransGradient, MIOPENConvGradientOp);
} // namespace caffe2
