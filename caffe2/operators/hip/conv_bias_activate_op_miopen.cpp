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

#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/operators/conv_op.h"
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
                  beta_(OperatorBase::GetSingleArgument<float>("beta", 0.0)) {
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bottom_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&bias_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&weight_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_));
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&top_desc_for_bias_));
            MIOPEN_ENFORCE(miopenCreateConvolutionDescriptor(&conv_desc_));

            if (group_ > 1) {
                mode_ = miopenGroupConv;
            } else {
                mode_ = miopenConvolution;
            }

            if (mode_ == miopenGroupConv) {
                OPERATOR_NEEDS_FEATURE(
                        dilation_h() == 1 && dilation_w() == 1,
                        "MIOpen convolution does not support dilation for groups > 1.");
            }
        }

        ~MIOPENConvBiasActivateOpBase() {
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
    };

    class MIOPENConvBiasActivateGradientOp final : public MIOPENConvBiasActivateOpBase {
    public:
        MIOPENConvBiasActivateGradientOp(const OperatorDef& operator_def, Workspace* ws)
                : MIOPENConvBiasActivateOpBase(operator_def, ws),
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

        ~MIOPENConvBiasActivateGradientOp() {
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
    bool MIOPENConvBiasActivateOp::DoRunWithType() {
        //TODO
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

    template <
            typename T_X,
            typename T_DY,
            typename T_W,
            typename T_B,
            typename MATH,
            typename T_DX,
            typename T_DW,
            typename T_DB>
    bool MIOPENConvBiasActivateGradientOp::DoRunWithType() {
        //TODO
        return true;
    }

    bool MIOPENConvBiasActivateGradientOp::RunOnDevice() {
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
        } else if (Input(0).IsType<at::Half>()) {
            return DoRunWithType<
                    at::Half, //  X
                    at::Half, // dY
                    at::Half, //  W
                    at::Half, //  b
                    at::Half, // Math
                    at::Half, // dX
                    at::Half, // dW
                    at::Half>(); // db
        } else {
            LOG(FATAL) << "Unsupported input types";
        }
        return true;
    }

    REGISTER_MIOPEN_OPERATOR(Conv, MIOPENConvBiasActivateOp);
    REGISTER_MIOPEN_OPERATOR(ConvGradient, MIOPENConvBiasActivateGradientOp);
} // namespace caffe2

