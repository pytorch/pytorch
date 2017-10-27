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

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default cudnn workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

// Manually specified number of algorithms implemented in CuDNN.
// This does not have any performance implications, as we will always find the
// fastest algorithm; setting them to the right number of algorithms will enable
// us to best report the statistics when doing an exhaustive search, though.
#if CUDNN_VERSION_MIN(7,0,0)
// Note: Double each of these due to potential
// tensorcode + non-tensorcore versions
// which are treated as seperate returned algos
static constexpr size_t kNUM_CUDNN_FWD_ALGS =
                                      2*CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS =
                                      2*CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS =
                                      2*CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
#else
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 7;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 4;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 5;
#endif

namespace {
template <typename ArrayOfcudnnConvolutionAlgoPerf_t>
inline void LogCuDNNPerfStats(
    const ArrayOfcudnnConvolutionAlgoPerf_t& perf_stat,
    int returned_algo_count) {
  VLOG(1) << "Perf result: (algo: stat, time, memory)";
  for (int i = 0; i < returned_algo_count; ++i) {
    const auto& stat = perf_stat[i];
    VLOG(1) << stat.algo << ": " << stat.status << " " << stat.time << " "
            << stat.memory;
  }
}

// Easier indexing into force_algo_ vector
enum {
  ALGO_FWD = 0,
  ALGO_WGRAD = 1,
  ALGO_DGRAD = 2
} algoIndex_t;

}  // namespace

class CudnnConvOpBase : public ConvPoolOpBase<CUDAContext> {
 public:
  CudnnConvOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        cudnn_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>(
            "ws_nbytes_limit",
            kCONV_CUDNN_WORKSPACE_LIMIT_BYTES)),
        exhaustive_search_(
            OperatorBase::GetSingleArgument<int>("exhaustive_search", 0)),
        deterministic_(
            OperatorBase::GetSingleArgument<int>("deterministic", 0)),
        cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0)),
        force_algo_(OperatorBase::GetRepeatedArgument<int>("force_algo", vector<int>{-1,-1,-1})),
        enable_tensor_core_(OperatorBase::GetSingleArgument<bool>("enable_tensor_core", 1)) {
    CHECK(!deterministic_ || !exhaustive_search_);
    CAFFE_ENFORCE(group_ > 0);
    CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);
    for (int i = 0; i < kernel_.size(); ++i) {
      OPERATOR_NEEDS_FEATURE(
          pads_[i] == pads_[kernel_.size() + i],
          "The current padding scheme leads to unequal padding on the left "
          "and right, which is not supported by cudnn.");
    }
    // dilated convolution supported by some algorithms in cuDNN v6
#if !(CUDNN_VERSION_MIN(6,0,0))
    OPERATOR_NEEDS_FEATURE(
        dilation_h() == 1 && dilation_w() == 1,
        "The cudnn convolution does not support dilation yet.");
#endif

    bool individual_force_algo = OperatorBase::HasArgument("force_algo_fwd") ||
                                 OperatorBase::HasArgument("force_algo_dgrad") ||
                                 OperatorBase::HasArgument("force_algo_wgrad");
    if (OperatorBase::HasArgument("force_algo")) {
      CAFFE_ENFORCE(!individual_force_algo,
                   "Cannot specify both force_algo and any of",
                   "force_algo_fwd, force_algo_dgrad, force_algo_wgrad");
    } else {
      force_algo_ = std::vector<int>{-1,-1,-1};
      force_algo_[ALGO_FWD] =
          OperatorBase::GetSingleArgument<int>("force_algo_fwd", -1);
      force_algo_[ALGO_DGRAD] =
          OperatorBase::GetSingleArgument<int>("force_algo_dgrad", -1);
      force_algo_[ALGO_WGRAD] =
          OperatorBase::GetSingleArgument<int>("force_algo_wgrad", -1);
    }

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
    CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CudnnConvOpBase() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:
  // A helper function to set up the tensor Nd desriptor, depending on the order
  // the group and the type given.
  template <typename T>
  void SetTensorNdDescriptorWithGroup(
      int size,
      cudnnTensorDescriptor_t desc_,
      int N,
      int C,
      int H,
      int W,
      int D) {
    switch (order_) {
      case StorageOrder::NHWC:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              desc_,
              cudnnTypeWrapper<T>::type,
              N,
#if CUDNN_VERSION_MIN(7,0,0)
              C,
#else
              C / group_,
#endif
              H,
              W,
              H * W * C,
              1,
              W * C,
              C));
        } else {
#if !CUDNN_VERSION_MIN(7,0,0)
          C = C / group_;
#endif
          vector<int> dims = {N, H, W, D, C};
          vector<int> strides = {H * W * D * C, W * D * C, D * C, C, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              desc_,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      case StorageOrder::NCHW:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              desc_,
              cudnnTypeWrapper<T>::type,
              N,
#if CUDNN_VERSION_MIN(7,0,0)
              C,
#else
              C / group_,
#endif
              H,
              W,
              C * H * W,
              H * W,
              W,
              1));
        } else {
#if !CUDNN_VERSION_MIN(7,0,0)
          C = C / group_;
#endif
          vector<int> dims = {N, C, H, W, D};
          vector<int> strides = {C * H * W * D, H * W * D, W * D, D, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              desc_,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
  }

  vector<TIndex> cudnn_input_dims_;
  vector<TIndex> cudnn_filter_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_;
  // top desc for bias add in case we do group convolution
  cudnnTensorDescriptor_t top_desc_for_bias_;
  cudnnConvolutionDescriptor_t conv_desc_;
  const size_t cudnn_ws_nbytes_limit_;
  size_t cudnn_ws_nbytes_;
  bool exhaustive_search_;
  bool deterministic_;
  size_t cudnn_state_;
  vector<int> force_algo_; // stored as FWD, dFILTER, dDATA
  bool enable_tensor_core_;
};


class CudnnConvOp final : public CudnnConvOpBase {
 public:
  CudnnConvOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws)  {}

  ~CudnnConvOp() {}

  template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> algo_cache_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

class CudnnConvGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  ~CudnnConvGradientOp() {}

  template <typename T_X, typename T_DY, typename T_W, typename T_B,
            typename MATH,
            typename T_DX, typename T_DW, typename T_DB>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t> filter_algo_cache_;
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t> data_algo_cache_;
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool CudnnConvOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);
  const int M = filter.dim32(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = Y->dim32(1);
      W_out = Y->ndim() > 3 ? Y->dim32(2) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = Y->dim32(2);
      W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }
      group_offset_X = C / group_ * H * W * D;
      group_offset_Y = M / group_ * H_out * W_out * D_out;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  int group_offset_filter = filter.size() / group_;

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensorNdDescriptorWithGroup<T_X>(
          X.ndim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
#if CUDNN_VERSION_MIN(7,0,0)
            M,
#else
            M / group_,
#endif
            C / group_,
            kernel_h(),
            kernel_w()));
      } else {
        vector<int> dims(filter.dims().begin(), filter.dims().end());
        dims[0] /= group_;
#if !CUDNN_VERSION_MIN(7,0,0)
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
#endif
        dims[filter.ndim() - 1] /= group_;
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (InputSize() == 3) {
        if (kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              bias_desc_,
              GetCudnnTensorFormat(order_),
              cudnnTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.ndim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              bias_desc_,
              cudnnTypeWrapper<T_B>::type,
              X.ndim() > 3 ? X.ndim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_Y>(
        X.ndim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T_B>::type,
          N,
          M,
          H_out,
          W_out));
    } else {
      vector<int> dims = {N, M, H_out, W_out, D_out};
      vector<int> strides = {M * H_out * W_out * D_out,
                             H_out * W_out * D_out,
                             H_out * D_out,
                             D_out,
                             1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<T_B>::type,
          X.ndim() > 3 ? X.ndim() : 4,
          dims.data(),
          strides.data()));
    }
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6,0,0)
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    } else {
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          dilation_.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#else
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          1,
          1,
          CUDNN_CROSS_CORRELATION));
    } else {
      vector<int> ones(dilation_.size(), 1);
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#endif

#if CUDNN_VERSION_MIN(7,0,0)
    // enable TensorCore math if desired
    enable_tensor_core_ &= TensorCoreAvailable();
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(cudnnSetConvolutionMathType(
            conv_desc_, CUDNN_TENSOR_OP_MATH));
    }

    // enable cuDNN conv groups
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
#endif

    if (force_algo_[ALGO_FWD] >= 0) {
      algo_ = (cudnnConvolutionFwdAlgo_t)force_algo_[ALGO_FWD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else if (exhaustive_search_) {
      algo_ = algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
        VLOG(1) << "CUDNN Convolution: doing exhaustive search.";
        // When we do an exhaustive search, we will ignore the workspace size
        // limit and simply go for the fastest algorithm. If you happen to run
        // out of memory later, you will be on your own...
        int returned_algo_count;
        std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
            perf_stat;

        // no need to clean up workspace,
        cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
          // Actually run the search.
          CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithmEx(
              state->cudnn_handle(),
              bottom_desc_,
              X.template data<T_X>(),
              filter_desc_,
              filter.template data<T_W>(),
              conv_desc_,
              top_desc_,
              Y->template mutable_data<T_Y>(),
              kNUM_CUDNN_FWD_ALGS,
              &returned_algo_count,
              perf_stat.data(),
              state->workspace().get(cudnn_ws_nbytes_limit_),
              cudnn_ws_nbytes_limit_));
        });
        LogCuDNNPerfStats(perf_stat, returned_algo_count);
        return perf_stat[0].algo;
      });
    } else {
      // Get the convolution algorithm based on the workspace limit.
      CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          filter_desc_,
          conv_desc_,
          top_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    CUDNN_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_,
        filter_desc_,
        conv_desc_,
        top_desc_,
        algo_,
        &cudnn_ws_nbytes_));
    VLOG(1) << "CuDNN algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  // Run directly through cuDNN if possible
#if CUDNN_VERSION_MIN(7,0,0)
  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnConvolutionForward(
        state->cudnn_handle(),
        cudnnTypeWrapper<T_X>::kOne(),
        bottom_desc_,
        X.template data<T_X>(),
        filter_desc_,
        filter.template data<T_W>(),
        conv_desc_,
        algo_,
        state->workspace().get(cudnn_ws_nbytes_),
        cudnn_ws_nbytes_,
        cudnnTypeWrapper<T_Y>::kZero(),
        top_desc_,
        Y->template mutable_data<T_Y>()));
  });
#else
  // otherwise manually run through groups
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnConvolutionForward(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_X>::kOne(),
          bottom_desc_,
          X.template data<T_X>() + i * group_offset_X,
          filter_desc_,
          filter.template data<T_W>() + i * group_offset_filter,
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_Y>::kZero(),
          top_desc_,
          Y->template mutable_data<T_Y>() + i * group_offset_Y));
    });
  }
#endif
  // Bias
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);

    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);

    CUDNN_ENFORCE(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_B>::kOne(),
        bias_desc_,
        bias.template data<T_B>(),
        cudnnTypeWrapper<T_Y>::kOne(),
        top_desc_for_bias_,
        Y->template mutable_data<T_Y>()));
  }
  // Done.
  return true;
}

bool CudnnConvOp::RunOnDevice() {

  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,      // X
                         float,      // W
                         float,      // B
                         float,      // Math
                         float>();   // Y
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,      // X
                         float16,      // W
                         float16,      // B
                         float,      // Math
                         float16>();   // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}

template <typename T_X, typename T_DY, typename T_W, typename T_B,
          typename MATH,
          typename T_DX, typename T_DW, typename T_DB>
bool CudnnConvGradientOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);

  const int M = filter.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = dY.dim32(1);
      W_out = dY.ndim() > 3 ? dY.dim32(2) : 1;
      D_out = dY.ndim() > 4 ? dY.dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = dY.dim32(2);
      W_out = dY.ndim() > 3 ? dY.dim32(3) : 1;
      D_out = dY.ndim() > 4 ? dY.dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }
      group_offset_X = C / group_ * H * W * D;
      group_offset_Y = M / group_ * H_out * W_out * D_out;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  int group_offset_filter = filter.size() / group_;
  if (kernel_.size() == 1) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H});
  } else if (kernel_.size() == 2) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W});
  } else if (kernel_.size() == 3) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W, D});
  } else {
    CAFFE_THROW("Unsupported kernel size:", kernel_.size());
  }
  dfilter->ResizeLike(filter);

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensorNdDescriptorWithGroup<T_X>(
          X.ndim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
#if CUDNN_VERSION_MIN(7,0,0)
            M,
#else
            M / group_,
#endif
            C / group_,
            kernel_h(),
            kernel_w()));
      } else {
        vector<int> dims(filter.dims().begin(), filter.dims().end());
#if !CUDNN_VERSION_MIN(7,0,0)
        dims[0] /= group_;
#endif
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (!no_bias_) {
        if (kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              bias_desc_,
              GetCudnnTensorFormat(order_),
              cudnnTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.ndim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              bias_desc_,
              cudnnTypeWrapper<T_B>::type,
              X.ndim() > 3 ? X.ndim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_DX>(
        X.ndim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T_B>::type,
          N,
          M,
          H_out,
          W_out));
    } else {
      vector<int> dims = {N, M, H_out, W_out, D_out};
      vector<int> strides = {M * H_out * W_out * D_out,
                             H_out * W_out * D_out,
                             H_out * D_out,
                             D_out,
                             1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<T_B>::type,
          X.ndim() > 3 ? X.ndim() : 4,
          dims.data(),
          strides.data()));
    }
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6,0,0)
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    } else {
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          dilation_.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#else
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          1,
          1,
          CUDNN_CROSS_CORRELATION));
    } else {
      vector<int> ones(dilation_.size(), 1);
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#endif

#if CUDNN_VERSION_MIN(7,0,0)
    // enable TensorCore math if desired
    enable_tensor_core_ &= TensorCoreAvailable();
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(cudnnSetConvolutionMathType(
            conv_desc_, CUDNN_TENSOR_OP_MATH));
    }

    // set cuDNN groups if appropriate
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
#endif

    // Set the workspace
    size_t bwd_filter_ws_size, bwd_data_ws_size;

    // Choose dW algorithm
    if (force_algo_[ALGO_WGRAD] >= 0) {
      bwd_filter_algo_ = (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
    } else if (deterministic_) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (exhaustive_search_) {
      bwd_filter_algo_ =
          filter_algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
            VLOG(1) << "CUDNN Convolution bwd: doing filter exhaustive search.";
            // When we do an exhaustive search, we will ignore the workspace
            // size
            // limit and simply go for the fastest algorithm. If you happen to
            // run
            // out of memory later, you will be on your own...
            int returned_algo_count;
            // We clean up the current workspace memory so that the forward
            // algorithm is free to allocate memory.
            // Actually run the search.
            std::array<
                cudnnConvolutionBwdFilterAlgoPerf_t,
                kNUM_CUDNN_BWD_FILTER_ALGS>
                filter_perf_stat;

            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                      state->cudnn_handle(),
                      bottom_desc_,
                      X.template data<T_X>(),
                      top_desc_,
                      dY.template data<T_DY>(),
                      conv_desc_,
                      filter_desc_,
                      dfilter->template mutable_data<T_DW>(),
                      kNUM_CUDNN_BWD_FILTER_ALGS,
                      &returned_algo_count,
                      filter_perf_stat.data(),
                      state->workspace().get(cudnn_ws_nbytes_limit_),
                      cudnn_ws_nbytes_limit_));
                });
            LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
            return filter_perf_stat[0].algo;
          });
    } else {
      // choose backward algorithm for filter
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          top_desc_,
          conv_desc_,
          filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &bwd_filter_algo_));
    }
    // Pick dX algo if needed
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      if (force_algo_[ALGO_DGRAD] >= 0) {
        bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
      } else if (deterministic_) {
        bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      } else if (exhaustive_search_) {
        bwd_data_algo_ =
            data_algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
              VLOG(1) << "CUDNN Convolution bwd: doing data exhaustive search.";
              int returned_algo_count;

              std::array<
                  cudnnConvolutionBwdDataAlgoPerf_t,
                  kNUM_CUDNN_BWD_DATA_ALGS>
                  data_perf_stat;
              cudnn_wrapper_.with_cudnn_state(
                  cudnn_state_, [&](CuDNNState* state) {
                    auto* dX =
                        Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
                    dX->ResizeLike(X);
                    const T_W* filter_data = filter.template data<T_W>();
                    const T_DY* dYdata = dY.template data<T_DY>();
                    T_DX* dXdata = dX->template mutable_data<T_DX>();
                    CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(
                        state->cudnn_handle(),
                        filter_desc_,
                        filter_data,
                        top_desc_,
                        dYdata,
                        conv_desc_,
                        bottom_desc_,
                        dXdata,
                        kNUM_CUDNN_BWD_DATA_ALGS,
                        &returned_algo_count,
                        data_perf_stat.data(),
                        state->workspace().get(cudnn_ws_nbytes_limit_),
                        cudnn_ws_nbytes_limit_));
                  });

              LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
              return data_perf_stat[0].algo;
            });
      } else {
        CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(
            cudnn_wrapper_.inline_cudnn_handle(),
            filter_desc_,
            top_desc_,
            conv_desc_,
            bottom_desc_,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            cudnn_ws_nbytes_limit_,
            &bwd_data_algo_));
      }
    }

    // get workspace for backwards filter algorithm
    CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_,
        top_desc_,
        conv_desc_,
        filter_desc_,
        bwd_filter_algo_,
        &bwd_filter_ws_size));
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // get workspace for backwards data algorithm
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          filter_desc_,
          top_desc_,
          conv_desc_,
          bottom_desc_,
          bwd_data_algo_,
          &bwd_data_ws_size));
    } else {
      bwd_data_ws_size = 0;
    }
    cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);

    VLOG(1) << "CuDNN bwd algorithm: " << bwd_filter_algo_ << ", "
            << bwd_data_algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_DY>::kOne(),
        top_desc_for_bias_,
        dY.template data<T_DY>(),
        cudnnTypeWrapper<T_DB>::kZero(),
        bias_desc_,
        dbias->template mutable_data<T_DB>()));
  }

#if CUDNN_VERSION_MIN(7,0,0)
  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
        state->cudnn_handle(),
        cudnnTypeWrapper<T_X>::kOne(),
        bottom_desc_,
        X.template data<T_X>(),
        top_desc_,
        dY.template data<T_DY>(),
        conv_desc_,
        bwd_filter_algo_,
        state->workspace().get(cudnn_ws_nbytes_),
        cudnn_ws_nbytes_,
        cudnnTypeWrapper<T_DW>::kZero(),
        filter_desc_,
        dfilter->template mutable_data<T_DW>()));
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // Compute the gradient w.r.t. the input.
      auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
      dX->ResizeLike(X);
      CUDNN_ENFORCE(cudnnConvolutionBackwardData(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_W>::kOne(),
          filter_desc_,
          filter.template data<T_W>(),
          top_desc_,
          dY.template data<T_DY>(),
          conv_desc_,
          bwd_data_algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_DX>::kZero(),
          bottom_desc_,
          dX->template mutable_data<T_DX>()));
    }
  });
#else
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_X>::kOne(),
          bottom_desc_,
          X.template data<T_X>() + i * group_offset_X,
          top_desc_,
          dY.template data<T_DY>() + i * group_offset_Y,
          conv_desc_,
          bwd_filter_algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_DW>::kZero(),
          filter_desc_,
          dfilter->template mutable_data<T_DW>() + i * group_offset_filter));
      if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
        // Compute the gradient w.r.t. the input.
        auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
        dX->ResizeLike(X);
        CUDNN_ENFORCE(cudnnConvolutionBackwardData(
            state->cudnn_handle(),
            cudnnTypeWrapper<T_W>::kOne(),
            filter_desc_,
            filter.template data<T_W>() + i * group_offset_filter,
            top_desc_,
            dY.template data<T_DY>() + i * group_offset_Y,
            conv_desc_,
            bwd_data_algo_,
            state->workspace().get(cudnn_ws_nbytes_),
            cudnn_ws_nbytes_,
            cudnnTypeWrapper<T_DX>::kZero(),
            bottom_desc_,
            dX->template mutable_data<T_DX>() + i * group_offset_X));
      }
    });
  }
#endif
  return true;
}

// TODO(Yangqing): a lot of the function contents are very similar. Consider
// consolidating them.
bool CudnnConvGradientOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,    //  X
                         float,    // dY
                         float,    //  W
                         float,    //  b
                         float,    // Math
                         float,    // dX
                         float,    // dW
                         float>(); // db
  }
  else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,    //  X
                         float16,    // dY
                         float16,    //  W
                         float16,    //  b
                         float,    // Math
                         float16,    // dX
                         float16,    // dW
                         float16>(); // db
  } else {
    LOG(FATAL) << "Unsupported input types";
  }
  return true;
}

REGISTER_CUDNN_OPERATOR(Conv, CudnnConvOp);
REGISTER_CUDNN_OPERATOR(ConvGradient, CudnnConvGradientOp);

REGISTER_CUDNN_OPERATOR(Conv1D, CudnnConvOp);
REGISTER_CUDNN_OPERATOR(Conv1DGradient, CudnnConvGradientOp);

REGISTER_CUDNN_OPERATOR(Conv2D, CudnnConvOp);
REGISTER_CUDNN_OPERATOR(Conv2DGradient, CudnnConvGradientOp);

REGISTER_CUDNN_OPERATOR(Conv3D, CudnnConvOp);
REGISTER_CUDNN_OPERATOR(Conv3DGradient, CudnnConvGradientOp);

}  // namespace caffe2
