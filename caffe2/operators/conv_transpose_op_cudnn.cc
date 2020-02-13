#include "caffe2/operators/conv_transpose_op.h"

#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/op_utils_cudnn.h"

namespace caffe2 {

class CudnnConvTransposeOpBase : public ConvTransposeUnpoolBase<CUDAContext> {
 public:
  template <class... Args>
  explicit CudnnConvTransposeOpBase(Args&&... args)
      : ConvTransposeUnpoolBase<CUDAContext>(std::forward<Args>(args)...),
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
    CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);

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
    if (InputSize() == 3) {
      CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
      CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
    }
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CudnnConvTransposeOpBase() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
    if (InputSize() == 3) {
      CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
      CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
    }
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:
  void SetTensor4DDescriptorWithGroup(
      const cudnnDataType_t data_type,
      const int N,
      const int C,
      const int H,
      const int W,
      cudnnTensorDescriptor_t* desc) const {
#if CUDNN_VERSION_MIN(7, 0, 0)
    const int CC = C;
#else
    const int CC = C / group_;
#endif
    switch (order_) {
      case StorageOrder::NCHW: {
        CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
            *desc, data_type, N, CC, H, W, C * H * W, H * W, W, 1));
        break;
      }
      case StorageOrder::NHWC: {
        CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
            *desc, data_type, N, CC, H, W, H * W * C, 1, W * C, C));
        break;
      }
      default: {
        LOG(FATAL) << "Unknown storage order: " << order_;
      }
    }
  }

  std::vector<std::int64_t> cudnn_input_dims_;
  std::vector<std::int64_t> cudnn_filter_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t top_desc_for_bias_;
  cudnnConvolutionDescriptor_t conv_desc_;

  const size_t cudnn_ws_nbytes_limit_;
  size_t cudnn_ws_nbytes_;
  bool exhaustive_search_;
  bool deterministic_;
  size_t cudnn_state_;
  std::vector<int> force_algo_; // stored as FWD, dFILTER, dDATA
  bool enable_tensor_core_;
};

template <typename T>
class CudnnConvTransposeOp final : public CudnnConvTransposeOpBase {
 public:
  template <class... Args>
  explicit CudnnConvTransposeOp(Args&&... args)
      : CudnnConvTransposeOpBase(std::forward<Args>(args)...) {}

  ~CudnnConvTransposeOp() override {}

  bool RunOnDevice() override;

 private:
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t> data_algo_cache_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T>
class CudnnConvTransposeGradientOp final : public CudnnConvTransposeOpBase {
 public:
  template <class... Args>
  explicit CudnnConvTransposeGradientOp(Args&&... args)
      : CudnnConvTransposeOpBase(std::forward<Args>(args)...),
        no_bias_(OperatorBase::GetSingleArgument<bool>("no_bias", false)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  ~CudnnConvTransposeGradientOp() override {}

  bool RunOnDevice() override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> forward_algo_cache_;
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t> filter_algo_cache_;
  const bool no_bias_;
  // input: X, W, dY
  // output: dW, optionally db and dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool CudnnConvTransposeOp<T>::RunOnDevice() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  int C = 0;
  switch (order_) {
    case StorageOrder::NHWC:
      C = filter.dim32(3) * group_;
      break;
    case StorageOrder::NCHW:
      C = filter.dim32(1) * group_;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  auto sizes = ConvTransposeUnpoolBase<CUDAContext>::GetOutputSize(X, C);
  auto* Y = Output(0, sizes, at::dtype<T>());

  if (X.numel() == 0) {
    VLOG(2) << "Number on elements is 0 in CudnnConvTransposeOp";
    return true;
  }

  int N = 0, M = 0, H = 0, W = 0, H_out = 0, W_out = 0;
  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.dim32(2);
      M = X.dim32(3);
      H_out = Y->dim32(1);
      W_out = Y->dim32(2);
      CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
      CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      M = X.dim32(1);
      H = X.dim32(2);
      W = X.dim32(3);
      H_out = Y->dim32(2);
      W_out = Y->dim32(3);
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w());
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }
  CAFFE_ENFORCE_EQ(M % group_, 0);

  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.dim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), C);
  }

  // Set up the cudnn algorithms & workspace if necessary
  const bool input_changed = (X.sizes() != cudnn_input_dims_);
  const bool filter_changed = (filter.sizes() != cudnn_filter_dims_);

  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.sizes().vec();
      SetTensor4DDescriptorWithGroup(
          cudnnTypeWrapper<T>::type, N, M, H, W, &bottom_desc_);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.sizes().vec();
#if CUDNN_VERSION_MIN(7, 0, 0)
      const int MM = M;
#else
      const int MM = M / group_;
#endif
      CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
          filter_desc_,
          cudnnTypeWrapper<T>::type,
          GetCudnnTensorFormat(order_),
          MM,
          C / group_,
          kernel_h(),
          kernel_w()));
      if (InputSize() == 3) {
        CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
            bias_desc_,
            GetCudnnTensorFormat(order_),
            cudnnTypeWrapper<T>::type,
            1,
            C,
            1,
            1));
      }
    }
    // Set the output
    SetTensor4DDescriptorWithGroup(
        cudnnTypeWrapper<T>::type, N, C, H_out, W_out, &top_desc_);
    if (InputSize() == 3) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          H_out,
          W_out));
    }

    // Set the convolution descriptor
    CAFFE_ENFORCE_EQ(
        pad_t(),
        pad_b(),
        "The current padding scheme leads to unequal padding on the top and "
        "bottom, which is not supported by cudnn.");
    CAFFE_ENFORCE_EQ(
        pad_l(),
        pad_r(),
        "The current padding scheme leads to unequal padding on the left "
        "and right, which is not supported by cudnn.");
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        cudnnTypeWrapper<T>::type));
#else
    CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        1,
        1,
        CUDNN_CROSS_CORRELATION));
#endif

#if CUDNN_VERSION_MIN(7, 0, 0)
    // enable TensorCore math if desired
    enable_tensor_core_ &= TensorCoreAvailable();
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    }
    // set cuDNN groups if appropriate
    CUDNN_ENFORCE(cudnnSetConvolutionGroupCount(conv_desc_, group_));
#endif

    if (force_algo_[ALGO_DGRAD] >= 0) {
      bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
    } else if (deterministic_) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else if (exhaustive_search_) {
      bwd_data_algo_ =
          data_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
            int returned_algo_count;
            std::array<
                cudnnConvolutionBwdDataAlgoPerf_t,
                kNUM_CUDNN_BWD_DATA_ALGS>
                data_perf_stat;
            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  state->workspace().reset();
                  CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithm(
                      state->cudnn_handle(),
                      filter_desc_,
                      bottom_desc_,
                      conv_desc_,
                      top_desc_,
                      kNUM_CUDNN_BWD_DATA_ALGS,
                      &returned_algo_count,
                      data_perf_stat.data()));
                });

            LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
            return data_perf_stat[0].algo;
          });
    } else {
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          filter_desc_,
          bottom_desc_,
          conv_desc_,
          top_desc_,
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &bwd_data_algo_));
    }

    size_t bwd_data_ws_size;
    CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        filter_desc_,
        bottom_desc_,
        conv_desc_,
        top_desc_,
        bwd_data_algo_,
        &bwd_data_ws_size));
    cudnn_ws_nbytes_ = bwd_data_ws_size;
    VLOG(1) << "CuDNN algorithm: " << bwd_data_algo_;
    VLOG(1) << "CuDNN workspace size: " << bwd_data_ws_size;
  }

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  T* Y_data = Y->template mutable_data<T>();

  // Now, actually run the computation.
  // Filter
#if CUDNN_VERSION_MIN(7, 0, 0)
  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnConvolutionBackwardData(
        state->cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        filter_desc_,
        filter_data,
        bottom_desc_,
        X_data,
        conv_desc_,
        bwd_data_algo_,
        state->workspace().get(cudnn_ws_nbytes_),
        cudnn_ws_nbytes_,
        cudnnTypeWrapper<T>::kZero(),
        top_desc_,
        Y_data));
  });
#else
  const int X_HxW = H * W;
  const int Y_HxW = H_out * W_out;
  const int group_offset_X =
      order_ == StorageOrder::NCHW ? M / group_ * X_HxW : M / group_;
  const int group_offset_Y =
      order_ == StorageOrder::NCHW ? C / group_ * Y_HxW : C / group_;
  const int group_offset_filter = filter.numel() / group_;
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(
          cudnnConvolutionBackwardData(state->cudnn_handle(),
                                       cudnnTypeWrapper<T>::kOne(),
                                       filter_desc_,
                                       filter_data + i * group_offset_filter,
                                       bottom_desc_,
                                       X_data + i * group_offset_X;
                                       conv_desc_,
                                       bwd_data_algo_,
                                       state->workspace().get(cudnn_ws_nbytes_),
                                       cudnn_ws_nbytes_,
                                       cudnnTypeWrapper<T_DX>::kZero(),
                                       top_desc_,
                                       Y_data + i * group_offset_Y));
    });
  }
#endif
  // Bias
  if (InputSize() == 3) {
    CUDNN_ENFORCE(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        bias_desc_,
        Input(BIAS).template data<T>(),
        cudnnTypeWrapper<T>::kOne(),
        top_desc_for_bias_,
        Y->template mutable_data<T>()));
  }
  // Done.
  return true;
}

// TODO(Yangqing): a lot of the function contents are very similar. Consider
// consolidating them.
template <typename T>
bool CudnnConvTransposeGradientOp<T>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  const auto& dY = Input(OUTPUT_GRAD);

  CAFFE_ENFORCE_EQ(X.dim(), 4);
  CAFFE_ENFORCE_EQ(filter.dim(), 4);
  int C = 0;
  switch (order_) {
    case StorageOrder::NHWC:
      C = filter.dim32(3) * group_;
      break;
    case StorageOrder::NCHW:
      C = filter.dim32(1) * group_;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  int N = 0, M = 0, H = 0, W = 0, H_out = 0, W_out = 0;
  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.dim32(2);
      M = X.dim32(3);
      H_out = dY.dim32(1);
      W_out = dY.dim32(2);
      CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
      CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      M = X.dim32(1);
      H = X.dim32(2);
      W = X.dim32(3);
      H_out = dY.dim32(2);
      W_out = dY.dim32(3);
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w());
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }
  CAFFE_ENFORCE_EQ(M % group_, 0);

  // Since we only handle LegacyPadding::NOTSET, we don't need to
  // compute padding.
  auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());

  // Set up the cudnn algorithms & workspace if necessary
  const bool input_changed = (X.sizes() != cudnn_input_dims_);
  const bool filter_changed = (filter.sizes() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.sizes().vec();
      SetTensor4DDescriptorWithGroup(
          cudnnTypeWrapper<T>::type, N, M, H, W, &bottom_desc_);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.sizes().vec();
#if CUDNN_VERSION_MIN(7, 0, 0)
      const int MM = M;
#else
      const int MM = M / group_;
#endif
      CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
          filter_desc_,
          cudnnTypeWrapper<T>::type,
          GetCudnnTensorFormat(order_),
          MM,
          C / group_,
          kernel_h(),
          kernel_w()));
      if (!no_bias_) {
        CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
            bias_desc_,
            GetCudnnTensorFormat(order_),
            cudnnTypeWrapper<T>::type,
            1,
            C,
            1,
            1));
      }
    }
    // Set the output
    SetTensor4DDescriptorWithGroup(
        cudnnTypeWrapper<T>::type, N, C, H_out, W_out, &top_desc_);
    if (!no_bias_) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          H_out,
          W_out));
    }

    // Set the convolution descriptor
    CAFFE_ENFORCE_EQ(
        pad_t(),
        pad_b(),
        "The current padding scheme leads to unequal padding on the top and "
        "bottom, which is not supported by cudnn.");
    CAFFE_ENFORCE_EQ(
        pad_l(),
        pad_r(),
        "The current padding scheme leads to unequal padding on the left "
        "and right, which is not supported by cudnn.");
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        cudnnTypeWrapper<T>::type));
#else
    CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        1,
        1,
        CUDNN_CROSS_CORRELATION));
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
    // enable TensorCore math if desired
    enable_tensor_core_ &= TensorCoreAvailable();
    if (enable_tensor_core_) {
      CUDNN_ENFORCE(
          cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    }
    // set cuDNN groups if appropriate
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
#endif
    if (force_algo_[ALGO_WGRAD] >= 0) {
      bwd_filter_algo_ =
          (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (exhaustive_search_) {
      bwd_filter_algo_ =
          filter_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
            LOG(INFO) << "CUDNN Convolution bwd: doing exhaustive search.";
            // When we do an exhaustive search, we will ignore the workspace
            // size
            // limit and simply go for the fastest algorithm. If you happen to
            // run
            // out of memory later, you will be on your own...
            int returned_algo_count;
            // We clean up the current workspace memory so that the forward
            // algorithm
            // is free to allocate memory.
            // Actually run the search.
            std::array<
                cudnnConvolutionBwdFilterAlgoPerf_t,
                kNUM_CUDNN_BWD_FILTER_ALGS>
                filter_perf_stat;

            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  state->workspace().reset();
                  CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithm(
                      state->cudnn_handle(),
                      top_desc_,
                      bottom_desc_,
                      conv_desc_,
                      filter_desc_,
                      kNUM_CUDNN_BWD_FILTER_ALGS,
                      &returned_algo_count,
                      filter_perf_stat.data()));
                });
            LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
            return filter_perf_stat[0].algo;
          });

      algo_ =
          forward_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
            int returned_algo_count;
            std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
                fwd_perf_stat;
            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  state->workspace().reset();
                  CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithm(
                      state->cudnn_handle(),
                      top_desc_,
                      filter_desc_,
                      conv_desc_,
                      bottom_desc_,
                      kNUM_CUDNN_BWD_DATA_ALGS,
                      &returned_algo_count,
                      fwd_perf_stat.data()));
                });

            LogCuDNNPerfStats(fwd_perf_stat, returned_algo_count);
            return fwd_perf_stat[0].algo;
          });
    } else {
      // choose backward algorithm for filter
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          top_desc_,
          bottom_desc_,
          conv_desc_,
          filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &bwd_filter_algo_));
      // choose backward algo for data
      CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          top_desc_,
          filter_desc_,
          conv_desc_,
          bottom_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    // get workspace for backwards filter algorithm
    size_t bwd_filter_ws_size, fwd_ws_size;
    CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        top_desc_,
        bottom_desc_,
        conv_desc_,
        filter_desc_,
        bwd_filter_algo_,
        &bwd_filter_ws_size));
    // get workspace for backwards data algorithm
    CUDNN_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        top_desc_,
        filter_desc_,
        conv_desc_,
        bottom_desc_,
        algo_,
        &fwd_ws_size));
    cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, fwd_ws_size);

    VLOG(1) << "CuDNN bwd algorithm: " << bwd_filter_algo_ << ", " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD, {C}, at::dtype<T>());
    CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        top_desc_for_bias_,
        dY.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        bias_desc_,
        dbias->template mutable_data<T>()));
  }

#if CUDNN_VERSION_MIN(7, 0, 0)
  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
        state->cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        top_desc_,
        dY.template data<T>(),
        bottom_desc_,
        X.template data<T>(),
        conv_desc_,
        bwd_filter_algo_,
        state->workspace().get(cudnn_ws_nbytes_),
        cudnn_ws_nbytes_,
        cudnnTypeWrapper<T>::kZero(),
        filter_desc_,
        dfilter->template mutable_data<T>()));

    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      // Compute the gradient w.r.t. the input.
      auto* dX = Output(
          no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
          X.sizes(),
          at::dtype<T>());
      CUDNN_ENFORCE(cudnnConvolutionForward(
          state->cudnn_handle(),
          cudnnTypeWrapper<T>::kOne(),
          top_desc_,
          dY.template data<T>(),
          filter_desc_,
          filter.template data<T>(),
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T>::kZero(),
          bottom_desc_,
          dX->template mutable_data<T>()));
    }
  });
#else
  const int X_HxW = H * W;
  const int Y_HxW = H_out * W_out;
  const int group_offset_X =
      order_ == StorageOrder::NCHW ? M / group_ * X_HxW : M / group_;
  const int group_offset_Y =
      order_ == StorageOrder::NCHW ? C / group_ * Y_HxW : C / group_;
  const int group_offset_filter = filter.numel() / group_;
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
          state->cudnn_handle(),
          cudnnTypeWrapper<T>::kOne(),
          top_desc_,
          dY.template data<T>() + i * group_offset_Y,
          bottom_desc_,
          X.template data<T>() + i * group_offset_X,
          conv_desc_,
          bwd_filter_algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T>::kZero(),
          filter_desc_,
          dfilter->template mutable_data<T>() + i * group_offset_filter));
      if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
        // Compute the gradient w.r.t. the input.
        auto* dX = Output(
            no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
            X.sizes(),
            at::dtype<T>());
        cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
          CUDNN_ENFORCE(cudnnConvolutionForward(
              state->cudnn_handle(),
              cudnnTypeWrapper<T>::kOne(),
              top_desc_,
              dY.template data<T>() + i * group_offset_Y,
              filter_desc_,
              filter.template data<T>() + i * group_offset_filter,
              conv_desc_,
              algo_,
              state->workspace().get(cudnn_ws_nbytes_),
              cudnn_ws_nbytes_,
              cudnnTypeWrapper<T>::kZero(),
              bottom_desc_,
              dX->template mutable_data<T>() + i * group_offset_X));
        });
      }
  }
#endif
  return true;
}

REGISTER_CUDNN_OPERATOR(ConvTranspose, CudnnConvTransposeOp<float>);
REGISTER_CUDNN_OPERATOR(
    ConvTransposeGradient,
    CudnnConvTransposeGradientOp<float>);

} // namespace caffe2
