#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default cudnn workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 64*1024*1024;

// Manually specified number of algorithms implemented in CuDNN.
// This does not have any performance implications, as we will always find the
// fastest algorithm; setting them to the right number of algorithms will enable
// us to best report the statistics when doing an exhaustive search, though.
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 7;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 4;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 5;

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
        cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0)) {
    CAFFE_ENFORCE(group_ > 0);
    CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);
    OPERATOR_NEEDS_FEATURE(
        pad_t_ == pad_b_,
        "The current padding scheme leads to unequal padding on the top and "
        "bottom, which is not supported by cudnn.");
    OPERATOR_NEEDS_FEATURE(
        pad_l_ == pad_r_,
        "The current padding scheme leads to unequal padding on the left "
        "and right, which is not supported by cudnn.");
    // dilated convolution supported by some algorithms in cuDNN v6
#if !(CUDNN_VERSION_MIN(6,0,0))
    OPERATOR_NEEDS_FEATURE(
        dilation_h_ == 1 && dilation_w_ == 1,
        "The cudnn convolution does not support dilation yet.");
#endif

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CudnnConvOpBase() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:
  // A helper function to set up the tensor 4d desriptor, depending on the order
  // the group and the type given.
  template <typename T>
  void SetTensor4dDescriptorWithGroup(
      cudnnTensorDescriptor_t desc_,
      int N,
      int C,
      int H,
      int W) {
    switch (order_) {
      case StorageOrder::NHWC:
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            desc_,
            cudnnTypeWrapper<T>::type,
            N,
            C / group_,
            H,
            W,
            H * W * C,
            1,
            W * C,
            C));
        break;
      case StorageOrder::NCHW:
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            desc_,
            cudnnTypeWrapper<T>::type,
            N,
            C / group_,
            H,
            W,
            C * H * W,
            H * W,
            W,
            1));
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
};

template <typename T>
class CudnnConvOp final : public CudnnConvOpBase {
 public:
  CudnnConvOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws)  {}

  ~CudnnConvOp() {}

  bool RunOnDevice() override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> algo_cache_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T>
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

template <typename T>
bool CudnnConvOp<T>::RunOnDevice() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

  // Figure out the output shape
  DCHECK_EQ(X.ndim(), 4);
  DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim32(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, H_out = 0, W_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  switch (order_) {
  case StorageOrder::NHWC:
    N = X.dim32(0); H = X.dim32(1); W = X.dim32(2); C = X.dim32(3);
    H_out = Y->dim32(1); W_out = Y->dim32(2);
    CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h_);
    CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w_);
    CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
    group_offset_X = C / group_;
    group_offset_Y = M / group_;
    break;
  case StorageOrder::NCHW:
    N = X.dim32(0); C = X.dim32(1); H = X.dim32(2); W = X.dim32(3);
    H_out = Y->dim32(2); W_out = Y->dim32(3);
    CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
    CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h_);
    CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w_);
    group_offset_X = C / group_ * H * W;
    group_offset_Y = M / group_ * H_out * W_out;
    break;
  default:
    LOG(FATAL) << "Unknown storage order: " << order_;
  }
  int group_offset_filter = filter.size() / group_;

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensor4dDescriptorWithGroup<T>(bottom_desc_, N, C, H, W);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(
          filter_desc_,
          cudnnTypeWrapper<T>::type,
          GetCudnnTensorFormat(order_),
          M / group_,
          C / group_,
          kernel_h_,
          kernel_w_));
      if (InputSize() == 3) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
              bias_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
              1, M, 1, 1));
      }
    }
    // Set the output
    SetTensor4dDescriptorWithGroup<T>(top_desc_, N, M, H_out, W_out);
    // Set the output with descriptor useful for bias addition in one run
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        top_desc_for_bias_,
        GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T>::type,
        N,
        M,
        H_out,
        W_out));
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6,0,0)
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, dilation_h_, dilation_w_,
          CUDNN_CROSS_CORRELATION, cudnnTypeWrapper<T>::type));
#else
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, 1, 1,
          CUDNN_CROSS_CORRELATION));
#endif
    if (deterministic_) {
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
          CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
              state->cudnn_handle(),
              bottom_desc_,
              X.template data<T>(),
              filter_desc_,
              filter.template data<T>(),
              conv_desc_,
              top_desc_,
              Y->template mutable_data<T>(),
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
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_, filter_desc_, conv_desc_, top_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_, filter_desc_, conv_desc_, top_desc_,
        algo_, &cudnn_ws_nbytes_));
    VLOG(1) << "CuDNN algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  // Filter
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_CHECK(cudnnConvolutionForward(
          state->cudnn_handle(),
          cudnnTypeWrapper<T>::kOne(),
          bottom_desc_,
          X.template data<T>() + i * group_offset_X,
          filter_desc_,
          filter.template data<T>() + i * group_offset_filter,
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T>::kZero(),
          top_desc_,
          Y->template mutable_data<T>() + i * group_offset_Y));
    });
  }
  // Bias
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);

    DCHECK_EQ(bias.ndim(), 1);
    DCHECK_EQ(bias.dim32(0), M);

    CUDNN_CHECK(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        bias_desc_,
        bias.template data<T>(),
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
bool CudnnConvGradientOp<T>::RunOnDevice() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);

  DCHECK_EQ(X.ndim(), 4);
  DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, H_out = 0, W_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  switch (order_) {
  case StorageOrder::NHWC:
    N = X.dim32(0); H = X.dim32(1); W = X.dim32(2); C = X.dim32(3);
    H_out = dY.dim32(1); W_out = dY.dim32(2);
    CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h_);
    CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w_);
    CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
    group_offset_X = C / group_;
    group_offset_Y = M / group_;
    break;
  case StorageOrder::NCHW:
    N = X.dim32(0); C = X.dim32(1); H = X.dim32(2); W = X.dim32(3);
    H_out = dY.dim32(2); W_out = dY.dim32(3);
    CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
    CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h_);
    CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w_);
    group_offset_X = C / group_ * H * W;
    group_offset_Y = M / group_ * H_out * W_out;
    break;
  default:
    LOG(FATAL) << "Unknown storage order: " << order_;
  }
  int group_offset_filter = filter.size() / group_;
  ConvPoolOpBase<CUDAContext>::ComputePads(H, W);
  dfilter->ResizeLike(filter);

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensor4dDescriptorWithGroup<T>(bottom_desc_, N, C, H, W);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(
          filter_desc_,
          cudnnTypeWrapper<T>::type,
          GetCudnnTensorFormat(order_),
          M / group_,
          C / group_,
          kernel_h_,
          kernel_w_));
      if (!no_bias_) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bias_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
            1, M, 1, 1));
      }
    }
    // Set the output
    SetTensor4dDescriptorWithGroup<T>(top_desc_, N, M, H_out, W_out);
    // Set the output with descriptor useful for bias addition in one run
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        top_desc_for_bias_,
        GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T>::type,
        N,
        M,
        H_out,
        W_out));
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6,0,0)
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, dilation_h_, dilation_w_,
          CUDNN_CROSS_CORRELATION, cudnnTypeWrapper<T>::type));
#else
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, 1, 1,
          CUDNN_CROSS_CORRELATION));
#endif
    // Set the workspace

    size_t bwd_filter_ws_size, bwd_data_ws_size;

    if (deterministic_) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
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
                  CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                      state->cudnn_handle(),
                      bottom_desc_,
                      X.template data<T>(),
                      top_desc_,
                      dY.template data<T>(),
                      conv_desc_,
                      filter_desc_,
                      dfilter->template mutable_data<T>(),
                      kNUM_CUDNN_BWD_FILTER_ALGS,
                      &returned_algo_count,
                      filter_perf_stat.data(),
                      state->workspace().get(cudnn_ws_nbytes_limit_),
                      cudnn_ws_nbytes_limit_));
                });
            LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
            return filter_perf_stat[0].algo;
          });

      if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
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
                    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
                        state->cudnn_handle(),
                        filter_desc_,
                        filter.template data<T>(),
                        top_desc_,
                        dY.template data<T>(),
                        conv_desc_,
                        bottom_desc_,
                        dX->template mutable_data<T>(),
                        kNUM_CUDNN_BWD_DATA_ALGS,
                        &returned_algo_count,
                        data_perf_stat.data(),
                        state->workspace().get(cudnn_ws_nbytes_limit_),
                        cudnn_ws_nbytes_limit_));
                  });

              LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
              return data_perf_stat[0].algo;
            });
      }
    } else {
      // choose backward algorithm for filter
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_, top_desc_, conv_desc_, filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_, &bwd_filter_algo_));
      // choose backward algo for data
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          filter_desc_, top_desc_, conv_desc_, bottom_desc_,
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_, &bwd_data_algo_));
    }
    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_, top_desc_, conv_desc_, filter_desc_,
        bwd_filter_algo_, &bwd_filter_ws_size));
    // get workspace for backwards data algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        filter_desc_, top_desc_, conv_desc_, bottom_desc_,
        bwd_data_algo_, &bwd_data_ws_size));
    cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);

    VLOG(1) << "CuDNN bwd algorithm: " << bwd_filter_algo_ << ", "
            << bwd_data_algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        top_desc_for_bias_,
        dY.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        bias_desc_,
        dbias->template mutable_data<T>()));
  }

  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          state->cudnn_handle(),
          cudnnTypeWrapper<T>::kOne(),
          bottom_desc_,
          X.template data<T>() + i * group_offset_X,
          top_desc_,
          dY.template data<T>() + i * group_offset_Y,
          conv_desc_,
          bwd_filter_algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T>::kZero(),
          filter_desc_,
          dfilter->template mutable_data<T>() + i * group_offset_filter));
      if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
        // Compute the gradient w.r.t. the input.
        auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
        dX->ResizeLike(X);
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            state->cudnn_handle(),
            cudnnTypeWrapper<T>::kOne(),
            filter_desc_,
            filter.template data<T>() + i * group_offset_filter,
            top_desc_,
            dY.template data<T>() + i * group_offset_Y,
            conv_desc_,
            bwd_data_algo_,
            state->workspace().get(cudnn_ws_nbytes_),
            cudnn_ws_nbytes_,
            cudnnTypeWrapper<T>::kZero(),
            bottom_desc_,
            dX->template mutable_data<T>() + i * group_offset_X));
      }
    });
  }
  return true;
}

REGISTER_CUDNN_OPERATOR(Conv, CudnnConvOp<float>);
REGISTER_CUDNN_OPERATOR(ConvGradient, CudnnConvGradientOp<float>);

REGISTER_CUDNN_OPERATOR(ConvFp16, CudnnConvOp<float16>);
REGISTER_CUDNN_OPERATOR(ConvFp16Gradient, CudnnConvGradientOp<float16>);

class GetConvFp16Gradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);
    if (def_.input_size() == 3) {
      return SingleGradientDef(
          "ConvFp16Gradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(2), GI(0)});
    } else {
      return SingleGradientDef(
          "ConvFp16Gradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(0)},
          vector<Argument>{MakeArgument<int>("no_bias", 1)});
    }
  }
};
REGISTER_GRADIENT(ConvFp16, GetConvFp16Gradient);

}  // namespace caffe2
