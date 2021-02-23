#include "caffe2/operators/conv_pool_op_base.h"

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/op_utils_cudnn.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

class CudnnConvOpBase : public ConvPoolOpBase<CUDAContext> {
 public:
  explicit CudnnConvOpBase(const OperatorDef& operator_def, Workspace* ws)
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
        force_algo_(OperatorBase::GetRepeatedArgument<int>(
            "force_algo",
            vector<int>{-1, -1, -1})),
        enable_tensor_core_(
            OperatorBase::GetSingleArgument<bool>("enable_tensor_core", 1)) {
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
#if !(CUDNN_VERSION_MIN(6, 0, 0))
    OPERATOR_NEEDS_FEATURE(
        dilation_h() == 1 && dilation_w() == 1,
        "The cudnn convolution does not support dilation yet.");
#endif
    // dilated grouped convolution supported in cuDNN v7.1
#if !(CUDNN_VERSION_MIN(7, 1, 0))
    if (group_ != 1) {
      for (int dim = 0; dim < kernel_.size(); ++dim) {
        OPERATOR_NEEDS_FEATURE(
            dilation_[dim] == 1,
            "When group is used, dilation should not be set at the same time.");
      }
    }
#endif

#if CUDNN_VERSION_MIN(7, 0, 0)
    // verify TensorCore math is supported
    enable_tensor_core_ &= TensorCoreAvailable();
#else
    enable_tensor_core_ = false;
#endif

    bool individual_force_algo = OperatorBase::HasArgument("force_algo_fwd") ||
        OperatorBase::HasArgument("force_algo_dgrad") ||
        OperatorBase::HasArgument("force_algo_wgrad");
    if (OperatorBase::HasArgument("force_algo")) {
      CAFFE_ENFORCE(
          !individual_force_algo,
          "Cannot specify both force_algo and any of",
          "force_algo_fwd, force_algo_dgrad, force_algo_wgrad");
    } else {
      force_algo_ = std::vector<int>{-1, -1, -1};
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

  ~CudnnConvOpBase() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:
  // A helper function to set up the tensor Nd descriptor, depending on the order
  // the group and the type given.
  template <typename T>
  void SetTensorNdDescriptorWithGroup(
      int size,
      cudnnTensorDescriptor_t tensorDesc,
      int N,
      int C,
      int H,
      int W,
      int D) {
#if CUDNN_VERSION_MIN(7, 0, 0)
    const int CC = C;
#else
    const int CC = C / group_;
#endif
    switch (order_) {
      case StorageOrder::NHWC:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              N,
              CC,
              H,
              W,
              H * W * C,
              1,
              W * C,
              C));
        } else {
          vector<int> dims = {N, H, W, D, CC};
          vector<int> strides = {H * W * D * CC, W * D * CC, D * CC, CC, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      case StorageOrder::NCHW:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              N,
              CC,
              H,
              W,
              C * H * W,
              H * W,
              W,
              1));
        } else {
          vector<int> dims = {N, CC, H, W, D};
          vector<int> strides = {CC * H * W * D, H * W * D, W * D, D, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              tensorDesc,
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

  void DuplicateConvDesc(
      cudnnConvolutionDescriptor_t input,
      size_t kernelDims,
      size_t dilationDims,
      cudnnConvolutionDescriptor_t copy) {
    if (kernelDims == 1 || kernelDims == 2) {
      cudnnConvolutionMode_t mode;
      cudnnDataType_t dataType;
      int pad_height = 0;
      int pad_width = 0;
      int stride_height = 0;
      int stride_width = 0;
      int dilation_height = 0;
      int dilation_width = 0;

#if CUDNN_VERSION_MIN(6, 0, 0)
      CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
          input,
          &pad_height,
          &pad_width,
          &stride_height,
          &stride_width,
          &dilation_height,
          &dilation_width,
          &mode,
          &dataType));
#else
      CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
          input,
          &pad_height,
          &pad_width,
          &stride_height,
          &stride_width,
          &dilation_height,
          &dilation_width,
          &mode));
#endif

#if CUDNN_VERSION_MIN(6, 0, 0)
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          copy,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          mode,
          dataType));
#else
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          copy,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          mode));
#endif
    } else {
      cudnnConvolutionMode_t mode;
      cudnnDataType_t dataType;
      int arrayLength = 0;
      vector<int> ones(dilationDims, 1);
      CUDNN_ENFORCE(cudnnGetConvolutionNdDescriptor(
          input,
          kernel_.size(),
          &arrayLength,
          pads_.data(),
          stride_.data(),
          ones.data(),
          &mode,
          &dataType));

      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          copy,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          mode,
          dataType));
    }
  }

  template <typename T>
  cudnnDataType_t DetermineComputeTypeFromInput(const T& X) {
    const cudaDeviceProp& prop = GetDeviceProperty(0);
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    if (X.template IsType<at::Half>()) {
      if (float16_compute_ && prop.major >= 6) {
        VLOG(1) << "CUDNN Convolution: float16_compute specified and "
                << "supported, input data is Half - using Half "
                << "compute.";
        computeType = CUDNN_DATA_HALF;
      } else if (float16_compute_) {
        VLOG(1) << "CUDNN Convolution: float16_compute specified but"
                << "not supported, input data is Half - using float32 "
                << "compute.";
      } else {
        VLOG(1) << "CUDNN Convolution: float16_compute not specified but "
                << "input data is Half - using float32 compute.";
      }
    } else {
      VLOG(1) << "CUDNN Convolution: using float32 compute.";
    }
    return computeType;
  }

  void SetConvDescFromArguments() {
#if CUDNN_VERSION_MIN(6, 0, 0)
    if (kernel_.size() == 1 || kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          kernel_.size() == 1 ? 0 : pad_l(),
          stride_h(),
          kernel_.size() == 1 ? 1 : stride_w(),
          dilation_h(),
          kernel_.size() == 1 ? 1 : dilation_w(),
          CUDNN_CROSS_CORRELATION,
          compute_type_));
    } else {
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          dilation_.data(),
          CUDNN_CROSS_CORRELATION,
          compute_type_));
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
          compute_type_));
    }
#endif
  }

  void SetConvDescComputeType(
      cudnnConvolutionDescriptor_t conv_desc,
      cudnnDataType_t math) {
    if (kernel_.size() == 2) {
      cudnnConvolutionMode_t mode;
      cudnnDataType_t dataType;
      int pad_height = 0;
      int pad_width = 0;
      int stride_height = 0;
      int stride_width = 0;
      int dilation_height = 0;
      int dilation_width = 0;

#if CUDNN_VERSION_MIN(6, 0, 0)
      CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
          conv_desc,
          &pad_height,
          &pad_width,
          &stride_height,
          &stride_width,
          &dilation_height,
          &dilation_width,
          &mode,
          &dataType));
#else
      CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
          conv_desc,
          &pad_height,
          &pad_width,
          &stride_height,
          &stride_width,
          &dilation_height,
          &dilation_width,
          &mode));
#endif

#if CUDNN_VERSION_MIN(6, 0, 0)
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          mode,
          math));
#else
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          mode));
#endif
    } else {
      cudnnConvolutionMode_t mode;
      cudnnDataType_t dataType;
      int arrayLength = 0;
      vector<int> ones(dilation_.size(), 1);
      CUDNN_ENFORCE(cudnnGetConvolutionNdDescriptor(
          conv_desc,
          kernel_.size(),
          &arrayLength,
          pads_.data(),
          stride_.data(),
          ones.data(),
          &mode,
          &dataType));

      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          mode,
          math));
    }
  }

  vector<int64_t> cudnn_input_dims_;
  vector<int64_t> cudnn_filter_dims_;

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
  cudnnDataType_t compute_type_;
};

class CudnnConvOp final : public CudnnConvOpBase {
 public:
  explicit CudnnConvOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws) {}

  ~CudnnConvOp() override {}

  template <typename T_X, typename T_W, typename T_B, typename T_Y>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  using ConvFwdAlgorithmWithCost = std::tuple<cudnnConvolutionFwdAlgo_t, float>;
  AlgorithmsCache<ConvFwdAlgorithmWithCost> algo_cache_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

class CudnnConvGradientOp final : public CudnnConvOpBase {
 public:
  explicit CudnnConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");

    CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&bwd_data_conv_desc_));
    CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&bwd_filter_conv_desc_));
  }

  ~CudnnConvGradientOp() override {
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(bwd_data_conv_desc_));
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(bwd_filter_conv_desc_));
  }

  template <
      typename T_X,
      typename T_DY,
      typename T_W,
      typename T_B,
      typename T_DX,
      typename T_DW,
      typename T_DB>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionDescriptor_t bwd_filter_conv_desc_;
  cudnnConvolutionDescriptor_t bwd_data_conv_desc_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  using ConvBwdFilterAlgorithmWithCost =
      std::tuple<cudnnConvolutionBwdFilterAlgo_t, float>;
  using ConvBwdDataAlgorithmWithCost =
      std::tuple<cudnnConvolutionBwdDataAlgo_t, float>;
  AlgorithmsCache<ConvBwdFilterAlgorithmWithCost> filter_algo_cache_;
  AlgorithmsCache<ConvBwdDataAlgorithmWithCost> data_algo_cache_;
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

static constexpr std::array<cudnnDataType_t, 2> kComputeTypesToTry = {
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF};
static constexpr std::array<const char*, 2> kComputePassNames = {
    "fp32 compute",
    "fp16 compute"};

template <typename T_X, typename T_W, typename T_B, typename T_Y>
bool CudnnConvOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);

  // Figure out the output shape
  CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
  CAFFE_ENFORCE(filter.dim() >= 3 && filter.dim() <= 5);
  const int M = filter.dim32(0);
  auto output_sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, M);
  auto* Y = Output(0, output_sizes, at::dtype<T_Y>());

  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.dim() > 3 ? X.dim32(2) : 1;
      D = X.dim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.dim() - 1);
      H_out = Y->dim32(1);
      W_out = Y->dim() > 3 ? Y->dim32(2) : 1;
      D_out = Y->dim() > 4 ? Y->dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.dim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.dim() > 3 ? X.dim32(3) : 1;
      D = X.dim() > 4 ? X.dim32(4) : 1;
      H_out = Y->dim32(2);
      W_out = Y->dim() > 3 ? Y->dim32(3) : 1;
      D_out = Y->dim() > 4 ? Y->dim32(4) : 1;
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

  if (N == 0) {
    Y->template mutable_data<T_Y>();
    return true;
  }

  int group_offset_filter = filter.numel() / group_;

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.sizes() != cudnn_input_dims_);
  bool filter_changed = (filter.sizes() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.sizes().vec();
      SetTensorNdDescriptorWithGroup<T_X>(X.dim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.sizes().vec();
      if (kernel_.size() == 1 || kernel_.size() == 2) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        const int MM = M;
#else
        const int MM = M / group_;
#endif
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            MM,
            C / group_,
            kernel_h(),
            kernel_.size() == 1 ? 1 : kernel_w()));
      } else {
        vector<int> dims(filter.sizes().begin(), filter.sizes().end());
#if !CUDNN_VERSION_MIN(7, 0, 0)
        // We only need to divide dims by group_ when CUDNN version < 7.0
        // see CUDA group convolution doc: https://fburl.com/dgj6dvpd
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
#endif
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (InputSize() == 3) {
        if (kernel_.size() == 1 || kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              bias_desc_,
              GetCudnnTensorFormat(order_),
              cudnnTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.dim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              bias_desc_,
              cudnnTypeWrapper<T_B>::type,
              X.dim() > 3 ? X.dim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_Y>(
        X.dim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 1 || kernel_.size() == 2) {
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
                             W_out * D_out,
                             D_out,
                             1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<T_B>::type,
          X.dim() > 3 ? X.dim() : 4,
          dims.data(),
          strides.data()));
    }

    compute_type_ = DetermineComputeTypeFromInput(X);
    SetConvDescFromArguments();

#if CUDNN_VERSION_MIN(7, 0, 0)
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    }

    // enable cuDNN conv groups
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
#endif

    if (force_algo_[ALGO_FWD] >= 0) {
      algo_ = (cudnnConvolutionFwdAlgo_t)force_algo_[ALGO_FWD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else if (exhaustive_search_) {
      // Even when FP16 compute is supported and requested, try FP32
      // because it may be faster. However, if FP32 compute is specified,
      // FP16 is not a suitable alternative - early out from the loop.
      std::array<ConvFwdAlgorithmWithCost, 2> algosToCompare;
      for (int i = 0; i < 2; i++) {
        SetConvDescComputeType(conv_desc_, kComputeTypesToTry[i]);

        algosToCompare[i] = algo_cache_.getAlgorithm(
            X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
              VLOG(1) << "CUDNN Convolution fwd: doing exhaustive "
                      << "search for " << kComputePassNames[i];
              // When we do an exhaustive search, we will ignore the workspace
              // size limit and simply go for the fastest algorithm. If you
              // happen to run out of memory later, you will be on your own...
              int returned_algo_count;
              std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
                  fwd_perf_stat;

              // no need to clean up workspace,
              cudnn_wrapper_.with_cudnn_state(
                  cudnn_state_, [&](CuDNNState* state) {
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
                        fwd_perf_stat.data(),
                        state->workspace().get(cudnn_ws_nbytes_limit_),
                        cudnn_ws_nbytes_limit_));
                  });
              LogCuDNNPerfStats(fwd_perf_stat, returned_algo_count);
              float algo_time = fwd_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                  ? fwd_perf_stat[0].time
                  : 1e10;
              return ConvFwdAlgorithmWithCost(fwd_perf_stat[0].algo, algo_time);
            });

        // When set to fp32 compute, don't try fp16
        if (compute_type_ == CUDNN_DATA_FLOAT) {
          break;
        }
      }

      if (compute_type_ == CUDNN_DATA_FLOAT) {
        // For FP32 compute, just use the best FP32 algorithm
        algo_ = std::get<0>(algosToCompare[0]);
      } else {
        // For FP16 compute, choose algo with fastest execution
        int bestAlgoIndex =
            (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
            ? 0
            : 1;
        algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
        SetConvDescComputeType(conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
      }
    } else {
      // Get the convolution algorithm based on the workspace limit.
      constexpr int nalgo = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
      int valid_algos;
      cudnnConvolutionFwdAlgoPerf_t algos[nalgo];
      CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm_v7(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          filter_desc_,
          conv_desc_,
          top_desc_,
          nalgo,
          &valid_algos,
          algos));
      bool found = false;
      for (int i = 0; i < valid_algos; i++) {
        auto a = algos[i];
        if (a.memory <= cudnn_ws_nbytes_limit_) {
          algo_ = a.algo;
          found = true;
          break;
        }
      }
      CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN forward");
    }
    for (int step = 0; step < 2; ++step) {
      cudnnStatus_t _status = cudnnGetConvolutionForwardWorkspaceSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          filter_desc_,
          conv_desc_,
          top_desc_,
          algo_,
          &cudnn_ws_nbytes_);
      if (step == 0) {
        if (_status == CUDNN_STATUS_SUCCESS) {
          break;
        }
        if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
          cudnnConvolutionFwdAlgo_t new_algo = deterministic_
              ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
              : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          VLOG(1) << "Forward algorithm " << (int)algo_
                  << " is not currently supported for given parameters."
                  << " Trying the default algorithm " << (int)new_algo;
          algo_ = new_algo;
          continue;
        }
      }
      CUDNN_ENFORCE(_status);
    }
    VLOG(1) << "CuDNN algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  // Run directly through cuDNN if possible
#if CUDNN_VERSION_MIN(7, 0, 0)
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

    CAFFE_ENFORCE_EQ(bias.dim(), 1);
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
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float>(); // Y
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<
        at::Half, // X
        at::Half, // W
        at::Half, // B
        at::Half>(); // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and Half are supported by "
               << "cudnn convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).dtype().name() << "]";
  }
  return true;
}

template <
    typename T_X,
    typename T_DY,
    typename T_W,
    typename T_B,
    typename T_DX,
    typename T_DW,
    typename T_DB>
bool CudnnConvGradientOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);

  CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
  CAFFE_ENFORCE(filter.dim() >= 3 && filter.dim() <= 5);

  const int M = filter.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.dim() > 3 ? X.dim32(2) : 1;
      D = X.dim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.dim() - 1);
      H_out = dY.dim32(1);
      W_out = dY.dim() > 3 ? dY.dim32(2) : 1;
      D_out = dY.dim() > 4 ? dY.dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.dim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.dim() > 3 ? X.dim32(3) : 1;
      D = X.dim() > 4 ? X.dim32(4) : 1;
      H_out = dY.dim32(2);
      W_out = dY.dim() > 3 ? dY.dim32(3) : 1;
      D_out = dY.dim() > 4 ? dY.dim32(4) : 1;
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

  int group_offset_filter = filter.numel() / group_;
  if (kernel_.size() == 1) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H});
  } else if (kernel_.size() == 2) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W});
  } else if (kernel_.size() == 3) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W, D});
  } else {
    CAFFE_THROW("Unsupported kernel size:", kernel_.size());
  }
  auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T_DW>());

  if (N == 0) {
    math::Set<T_DW, CUDAContext>(
        dfilter->numel(),
        T_DW(0),
        dfilter->template mutable_data<T_DW>(),
        &context_);
    if (!no_bias_) {
      auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T_DB>());
      math::Set<T_DB, CUDAContext>(
          dbias->numel(),
          T_DB(0),
          dbias->template mutable_data<T_DB>(),
          &context_);
    }
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      auto* dX = Output(
          no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
          X.sizes(),
          at::dtype<T_DX>());
      dX->template mutable_data<T_DX>();
    }
    return true;
  }

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.sizes() != cudnn_input_dims_);
  bool filter_changed = (filter.sizes() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.sizes().vec();
      SetTensorNdDescriptorWithGroup<T_X>(X.dim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.sizes().vec();
      if (kernel_.size() == 1 || kernel_.size() == 2) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        const int MM = M;
#else
        const int MM = M / group_;
#endif
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            MM,
            C / group_,
            kernel_h(),
            kernel_.size() == 1 ? 1 : kernel_w()));
      } else {
        vector<int> dims(filter.sizes().begin(), filter.sizes().end());
#if !CUDNN_VERSION_MIN(7, 0, 0)
        // We only need to divide dims by group_ when CUDNN version < 7.0
        // see CUDA group convolution doc: https://fburl.com/dgj6dvpd
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
#endif

        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (!no_bias_) {
        if (kernel_.size() == 1 || kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              bias_desc_,
              GetCudnnTensorFormat(order_),
              cudnnTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.dim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              bias_desc_,
              cudnnTypeWrapper<T_B>::type,
              X.dim() > 3 ? X.dim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_DX>(
        X.dim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 1 || kernel_.size() == 2) {
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
                             W_out * D_out,
                             D_out,
                             1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<T_B>::type,
          X.dim() > 3 ? X.dim() : 4,
          dims.data(),
          strides.data()));
    }

    compute_type_ = DetermineComputeTypeFromInput(X);
    SetConvDescFromArguments();

    DuplicateConvDesc(
        conv_desc_, kernel_.size(), dilation_.size(), bwd_filter_conv_desc_);
    DuplicateConvDesc(
        conv_desc_, kernel_.size(), dilation_.size(), bwd_data_conv_desc_);

#if CUDNN_VERSION_MIN(7, 0, 0)
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(cudnnSetConvolutionMathType(
          bwd_filter_conv_desc_, CUDNN_TENSOR_OP_MATH));
      CUDNN_ENFORCE(cudnnSetConvolutionMathType(
          bwd_data_conv_desc_, CUDNN_TENSOR_OP_MATH));
    }

    // set cuDNN groups if appropriate
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(bwd_filter_conv_desc_, group_));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(bwd_data_conv_desc_, group_));
#endif

    // Choose dW algorithm
    if (force_algo_[ALGO_WGRAD] >= 0) {
      bwd_filter_algo_ =
          (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
    } else if (deterministic_) {
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (exhaustive_search_) {
      // Even when FP16 compute is supported and requested, try FP32
      // because it may be faster. However, if FP32 compute is specified,
      // FP16 is not a suitable alternative - early out from the loop.
      std::array<ConvBwdFilterAlgorithmWithCost, 2> algosToCompare;
      for (int i = 0; i < 2; i++) {
        SetConvDescComputeType(bwd_filter_conv_desc_, kComputeTypesToTry[i]);

        algosToCompare[i] = filter_algo_cache_.getAlgorithm(
            X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
              VLOG(1) << "CUDNN Convolution bwd: doing filter exhaustive"
                      << "search for " << kComputePassNames[i];
              // When we do an exhaustive search, we will ignore the workspace
              // size limit and simply go for the fastest algorithm. If you
              // happen to run out of memory later, you will be on your own...
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
                        bwd_filter_conv_desc_,
                        filter_desc_,
                        dfilter->template mutable_data<T_DW>(),
                        kNUM_CUDNN_BWD_FILTER_ALGS,
                        &returned_algo_count,
                        filter_perf_stat.data(),
                        state->workspace().get(cudnn_ws_nbytes_limit_),
                        cudnn_ws_nbytes_limit_));
                  });
              LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
              float algo_time =
                  filter_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                  ? filter_perf_stat[0].time
                  : 1e10;
              return ConvBwdFilterAlgorithmWithCost(
                  filter_perf_stat[0].algo, algo_time);
            });

        // When set to fp32 compute, don't try fp16
        if (compute_type_ == CUDNN_DATA_FLOAT) {
          break;
        }
      }

      if (compute_type_ == CUDNN_DATA_FLOAT) {
        // For FP32 compute, just use the best FP32 algorithm
        bwd_filter_algo_ = std::get<0>(algosToCompare[0]);
      } else {
        // For FP16 compute, choose algo with fastest execution
        int bestAlgoIndex =
            (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
            ? 0
            : 1;
        bwd_filter_algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
        SetConvDescComputeType(
            bwd_filter_conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
      }
    } else {
      // choose backward algorithm for filter
      constexpr int nalgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
      int valid_algos;
      cudnnConvolutionBwdFilterAlgoPerf_t algos[nalgo];
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          top_desc_,
          bwd_filter_conv_desc_,
          filter_desc_,
          nalgo,
          &valid_algos,
          algos));
      bool found = false;
      for (int i = 0; i < valid_algos; i++) {
        auto a = algos[i];
        if (a.memory <= cudnn_ws_nbytes_limit_) {
          bwd_filter_algo_ = a.algo;
          found = true;
          break;
        }
      }
      CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN backward filter");
    }
    // Pick dX algo if needed
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      if (force_algo_[ALGO_DGRAD] >= 0) {
        bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
      } else if (deterministic_) {
        bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      } else if (exhaustive_search_) {
        // Even when FP16 compute is supported and requested, try FP32
        // because it may be faster. However, if FP32 compute is specified,
        // FP16 is not a suitable alternative - early out from the loop.
        std::array<ConvBwdDataAlgorithmWithCost, 2> algosToCompare;
        for (int i = 0; i < 2; i++) {
          SetConvDescComputeType(bwd_data_conv_desc_, kComputeTypesToTry[i]);

          algosToCompare[i] = data_algo_cache_.getAlgorithm(
              X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
                VLOG(1) << "CUDNN Convolution bwd: doing data exhaustive"
                        << "search for " << kComputePassNames[i];
                int returned_algo_count;

                std::array<
                    cudnnConvolutionBwdDataAlgoPerf_t,
                    kNUM_CUDNN_BWD_DATA_ALGS>
                    data_perf_stat;
                cudnn_wrapper_.with_cudnn_state(
                    cudnn_state_, [&](CuDNNState* state) {
                      auto* dX = Output(
                          no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                          X.sizes(),
                          at::dtype<T_DX>());
                      const T_W* filter_data = filter.template data<T_W>();
                      const T_DY* dYdata = dY.template data<T_DY>();
                      T_DX* dXdata = dX->template mutable_data<T_DX>();
                      CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(
                          state->cudnn_handle(),
                          filter_desc_,
                          filter_data,
                          top_desc_,
                          dYdata,
                          bwd_data_conv_desc_,
                          bottom_desc_,
                          dXdata,
                          kNUM_CUDNN_BWD_DATA_ALGS,
                          &returned_algo_count,
                          data_perf_stat.data(),
                          state->workspace().get(cudnn_ws_nbytes_limit_),
                          cudnn_ws_nbytes_limit_));
                    });

                LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
                float algo_time =
                    data_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                    ? data_perf_stat[0].time
                    : 1e10;
                return ConvBwdDataAlgorithmWithCost(
                    data_perf_stat[0].algo, algo_time);
              });

          // When set to fp32 compute, don't try fp16
          if (compute_type_ == CUDNN_DATA_FLOAT) {
            break;
          }
        }

        if (compute_type_ == CUDNN_DATA_FLOAT) {
          // For FP32 compute, just use the best FP32 algorithm
          bwd_data_algo_ = std::get<0>(algosToCompare[0]);
        } else {
          // For FP16 compute, choose algo with fastest execution
          int bestAlgoIndex =
              (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
              ? 0
              : 1;
          bwd_data_algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
          SetConvDescComputeType(
              bwd_data_conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
        }
      } else {
        constexpr int nalgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        int valid_algos;
        cudnnConvolutionBwdDataAlgoPerf_t algos[nalgo];
        CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            cudnn_wrapper_.inline_cudnn_handle(),
            filter_desc_,
            top_desc_,
            bwd_data_conv_desc_,
            bottom_desc_,
            nalgo,
            &valid_algos,
            algos));
        bool found = false;
        for (int i = 0; i < valid_algos; i++) {
          auto a = algos[i];
          if (a.memory <= cudnn_ws_nbytes_limit_) {
            bwd_data_algo_ = a.algo;
            found = true;
            break;
          }
        }
        CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN backward data");
      }
    }

    // get workspace size for backwards filter algorithm
    size_t bwd_filter_ws_size, bwd_data_ws_size;

    for (int step = 0; step < 2; ++step) {
      cudnnStatus_t _status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          top_desc_,
          bwd_filter_conv_desc_,
          filter_desc_,
          bwd_filter_algo_,
          &bwd_filter_ws_size);
      if (step == 0) {
        if (_status == CUDNN_STATUS_SUCCESS) {
          break;
        }
        if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
          cudnnConvolutionBwdFilterAlgo_t new_algo = deterministic_
              ? CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
              : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
          VLOG(1) << "Backward Filter algorithm " << (int)bwd_filter_algo_
                  << " is not currently supported for given parameters."
                  << " Trying the default algorithm " << (int)new_algo;
          bwd_filter_algo_ = new_algo;
          continue;
        }
      }
      CUDNN_ENFORCE(_status);
    }

    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // get workspace size for backwards data algorithm
      for (int step = 0; step < 2; ++step) {
        cudnnStatus_t _status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_wrapper_.inline_cudnn_handle(),
            filter_desc_,
            top_desc_,
            bwd_data_conv_desc_,
            bottom_desc_,
            bwd_data_algo_,
            &bwd_data_ws_size);
        if (step == 0) {
          if (_status == CUDNN_STATUS_SUCCESS) {
            break;
          }
          if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
            cudnnConvolutionBwdDataAlgo_t new_algo = deterministic_
                ? CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
                : CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            VLOG(1) << "Backward Data algorithm " << (int)bwd_data_algo_
                    << " is not currently supported for given parameters."
                    << " Trying the default algorithm " << (int)new_algo;
            bwd_data_algo_ = new_algo;
            continue;
          }
        }
        CUDNN_ENFORCE(_status);
      }
    } else {
      bwd_data_ws_size = 0;
    }
    cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);

    VLOG(1) << "CuDNN bwd data & filter algorithm: " << bwd_data_algo_ << ", "
            << bwd_filter_algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T_DB>());
    CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_DY>::kOne(),
        top_desc_for_bias_,
        dY.template data<T_DY>(),
        cudnnTypeWrapper<T_DB>::kZero(),
        bias_desc_,
        dbias->template mutable_data<T_DB>()));
  }

#if CUDNN_VERSION_MIN(7, 0, 0)
  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
        state->cudnn_handle(),
        cudnnTypeWrapper<T_X>::kOne(),
        bottom_desc_,
        X.template data<T_X>(),
        top_desc_,
        dY.template data<T_DY>(),
        bwd_filter_conv_desc_,
        bwd_filter_algo_,
        state->workspace().get(cudnn_ws_nbytes_),
        cudnn_ws_nbytes_,
        cudnnTypeWrapper<T_DW>::kZero(),
        filter_desc_,
        dfilter->template mutable_data<T_DW>()));
    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // Compute the gradient w.r.t. the input.

      auto* dX = Output(
          no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
          X.sizes(),
          at::dtype<T_DX>());
      CUDNN_ENFORCE(cudnnConvolutionBackwardData(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_W>::kOne(),
          filter_desc_,
          filter.template data<T_W>(),
          top_desc_,
          dY.template data<T_DY>(),
          bwd_data_conv_desc_,
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
          bwd_filter_conv_desc_,
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
            bwd_data_conv_desc_,
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
    return DoRunWithType<
        float, //  X
        float, // dY
        float, //  W
        float, //  b
        float, // dX
        float, // dW
        float>(); // db
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<
        at::Half, //  X
        at::Half, // dY
        at::Half, //  W
        at::Half, //  b
        at::Half, // dX
        at::Half, // dW
        at::Half>(); // db
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

} // namespace caffe2
