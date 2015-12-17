#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 8*1024*1024;

class CudnnConvOpBase : public ConvPoolOpBase<CUDAContext> {
 public:
  CudnnConvOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_),
        cudnn_ws_nbytes_limit_(
            OperatorBase::GetSingleArgument<int>(
                "ws_nbytes_limit", kCONV_CUDNN_WORKSPACE_LIMIT_BYTES)),
        shared_ws_name_(
            OperatorBase::GetSingleArgument<string>("shared_ws_name", "")),
        exhaustive_search_(
            OperatorBase::GetSingleArgument<int>("exhaustive_search", 0)) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    if (shared_ws_name_.size()) {
      // We will use a shared workspace for cudnn across multiple operators,
      // which would allow us to save memory space better.
      // Note that this is kind of a hack: the computation logic of the shared
      // workspace is not visible to the compute graph, so use this with care.
      // You are essentially responsible for managing potential conflicts of the
      // shared workspace yourself, and you need to make sure that this name
      // does not conflict with some other blob names in the compute graph.
      cudnn_ws_ = ws->CreateBlob(shared_ws_name_)
          ->GetMutable<CuDNNWorkspaceWrapper>();
    } else {
      // We will maintain a local workspace.
      local_cudnn_ws_.reset(new CuDNNWorkspaceWrapper());
      cudnn_ws_ = local_cudnn_ws_.get();
    }
  }

  ~CudnnConvOpBase() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  virtual bool RunWithCudnnWorkspace(
      CuDNNWorkspaceWrapper* cudnn_ws_wrapper) = 0;

  bool RunOnDevice() final {
    std::lock_guard<std::mutex> lock(cudnn_ws_->mutex());
    bool success = RunWithCudnnWorkspace(cudnn_ws_);
    // Since we will release the lock guard, we need to finish all the
    // device computation explicitly so the cudnn execution finishes.
    success &= device_context_.FinishDeviceComputation();
    return success;
  }

 protected:
  vector<int> cudnn_input_dims_;
  vector<int> cudnn_filter_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  const size_t cudnn_ws_nbytes_limit_;
  string shared_ws_name_;
  size_t cudnn_ws_nbytes_;
  CuDNNWorkspaceWrapper* cudnn_ws_;
  bool exhaustive_search_;
  std::unique_ptr<CuDNNWorkspaceWrapper> local_cudnn_ws_;
  DISABLE_COPY_AND_ASSIGN(CudnnConvOpBase);
};

template <typename T>
class CudnnConvOp final : public CudnnConvOpBase {
 public:
  CudnnConvOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws)  {}

  ~CudnnConvOp() {}

  bool RunWithCudnnWorkspace(CuDNNWorkspaceWrapper* cudnn_ws_wrapper) override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CudnnConvOp);
};

template <typename T>
class CudnnConvGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws)  {}

  ~CudnnConvGradientOp() {}

  bool RunWithCudnnWorkspace(CuDNNWorkspaceWrapper* cudnn_ws_wrapper) override;

 private:
#if CUDNN_VERSION >= 3000
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
#endif  // CUDNN_VERSION >= 3000
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_GRAD, INPUT_GRAD);
  INPUT_OUTPUT_STATS(3, 3, 2, 3);
  DISABLE_COPY_AND_ASSIGN(CudnnConvGradientOp);
};



////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool CudnnConvOp<T>::RunWithCudnnWorkspace(
      CuDNNWorkspaceWrapper* cudnn_ws_wrapper) {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& bias = Input(BIAS);
  auto* Y = Output(0);

  // Figure out the output shape
  CAFFE_DCHECK_EQ(X.ndim(), 4);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N, C, H, W, H_out, W_out;
  switch (order_) {
  case StorageOrder::NHWC:
    N = X.dim(0); H = X.dim(1); W = X.dim(2); C = X.dim(3);
    H_out = Y->dim(1); W_out = Y->dim(2);
    CAFFE_DCHECK_EQ(filter.dim(1), kernel_h_);
    CAFFE_DCHECK_EQ(filter.dim(2), kernel_w_);
    CAFFE_DCHECK_EQ(filter.dim(3), C);
    break;
  case StorageOrder::NCHW:
    N = X.dim(0); C = X.dim(1); H = X.dim(2); W = X.dim(3);
    H_out = Y->dim(2); W_out = Y->dim(3);
    CAFFE_DCHECK_EQ(filter.dim(1), C);
    CAFFE_DCHECK_EQ(filter.dim(2), kernel_h_);
    CAFFE_DCHECK_EQ(filter.dim(3), kernel_w_);
    break;
  default:
    CAFFE_LOG_FATAL << "Unknown storage order: " << order_;
  }
  CAFFE_DCHECK_EQ(bias.ndim(), 1);
  CAFFE_DCHECK_EQ(bias.dim(0), M);

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    CAFFE_VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          bottom_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          N, C, H, W));
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(
          filter_desc_, cudnnTypeWrapper<T>::type, M, C, kernel_h_, kernel_w_));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          bias_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          1, M, 1, 1));
    }
    // Set the output
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          top_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          N, M, H_out, W_out));
    // Set the convolution descriptor
    CAFFE_CHECK_EQ(pad_t_, pad_b_)
        << "The current padding scheme leads to unequal padding on the top and "
           "bottom, which is not supported by cudnn.";
    CAFFE_CHECK_EQ(pad_l_, pad_r_)
        << "The current padding scheme leads to unequal padding on the left "
           "and right, which is not supported by cudnn.";
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, 1, 1,
          CUDNN_CROSS_CORRELATION));
    if (exhaustive_search_) {
      CAFFE_VLOG(1) << "CUDNN Convolution: doing exhaustive search.";
      // When we do an exhaustive search, we will ignore the workspace size
      // limit and simply go for the fastest algorithm. If you happen to run
      // out of memory later, you will be on your own...
      int returned_algo_count;
      cudnnConvolutionFwdAlgoPerf_t perf_stat;
      // We clean up the current workspace memory so that the forward algorithm
      // is free to allocate memory.
      cudnn_ws_wrapper->Reset();
      // Actually run the search.
      CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          bottom_desc_, filter_desc_, conv_desc_, top_desc_,
          1, &returned_algo_count, &perf_stat));
      CAFFE_DCHECK_EQ(returned_algo_count, 1);
      algo_ = perf_stat.algo;
    } else {
      // Get the convolution algorithm based on the workspace limit.
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          bottom_desc_, filter_desc_, conv_desc_, top_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_wrapper_.cudnn_handle(),
        bottom_desc_, filter_desc_, conv_desc_, top_desc_,
        algo_, &cudnn_ws_nbytes_));
    CAFFE_VLOG(1) << "CuDNN algorithm: " << algo_;
    CAFFE_VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
  const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
  // Filter
  CUDNN_CHECK(cudnnConvolutionForward(
      cudnn_wrapper_.cudnn_handle(), &kOne, bottom_desc_,
      X.template data<T>(), filter_desc_, filter.template data<T>(), conv_desc_,
      algo_, cudnn_ws_wrapper->Get(cudnn_ws_nbytes_), cudnn_ws_nbytes_, &kZero,
      top_desc_, Y->template mutable_data<T>()));
  // Bias
#if CUDNN_VERSION >= 3000
  CUDNN_CHECK(cudnnAddTensor_v3(
      cudnn_wrapper_.cudnn_handle(), &kOne, bias_desc_,
      bias.template data<T>(), &kOne, top_desc_, Y->template mutable_data<T>()));
#else  // CUDNN_VERSION >= 3000
  CUDNN_CHECK(cudnnAddTensor(
      cudnn_wrapper_.cudnn_handle(), CUDNN_ADD_SAME_C,
      &kOne, bias_desc_, bias.template data<T>(), &kOne, top_desc_,
      Y->template mutable_data<T>()));
#endif
  // Done.
  return true;
}

// TODO(Yangqing): a lot of the function contents are very similar. Consider
// consolidating them.
template <typename T>
bool CudnnConvGradientOp<T>::RunWithCudnnWorkspace(
    CuDNNWorkspaceWrapper* cudnn_ws_wrapper) {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  auto* dbias = Output(BIAS_GRAD);
  CAFFE_DCHECK_EQ(X.ndim(), 4);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  int N, C, H, W, H_out, W_out;
  switch (order_) {
  case StorageOrder::NHWC:
    N = X.dim(0); H = X.dim(1); W = X.dim(2); C = X.dim(3);
    H_out = dY.dim(1); W_out = dY.dim(2);
    CAFFE_DCHECK_EQ(filter.dim(1), kernel_h_);
    CAFFE_DCHECK_EQ(filter.dim(2), kernel_w_);
    CAFFE_DCHECK_EQ(filter.dim(3), C);
    break;
  case StorageOrder::NCHW:
    N = X.dim(0); C = X.dim(1); H = X.dim(2); W = X.dim(3);
    H_out = dY.dim(2); W_out = dY.dim(3);
    CAFFE_DCHECK_EQ(filter.dim(1), C);
    CAFFE_DCHECK_EQ(filter.dim(2), kernel_h_);
    CAFFE_DCHECK_EQ(filter.dim(3), kernel_w_);
    break;
  default:
    CAFFE_LOG_FATAL << "Unknown storage order: " << order_;
  }
  ConvPoolOpBase<CUDAContext>::ComputePads(H, W);
  dfilter->ReshapeLike(filter);
  dbias->Reshape(std::vector<int>{M});

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    CAFFE_VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          bottom_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          N, C, H, W));
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(
          filter_desc_, cudnnTypeWrapper<T>::type, M, C, kernel_h_, kernel_w_));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          bias_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          1, M, 1, 1));
    }
    // Set the output
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          top_desc_, GetCudnnTensorFormat(order_), cudnnTypeWrapper<T>::type,
          N, M, H_out, W_out));
    // Set the convolution descriptor
    CAFFE_CHECK_EQ(pad_t_, pad_b_)
        << "The current padding scheme leads to unequal padding on the top and "
           "bottom, which is not supported by cudnn.";
    CAFFE_CHECK_EQ(pad_l_, pad_r_)
        << "The current padding scheme leads to unequal padding on the left "
           "and right, which is not supported by cudnn.";
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_, pad_t_, pad_l_, stride_h_, stride_w_, 1, 1,
          CUDNN_CROSS_CORRELATION));
    // Set the workspace

    size_t bwd_filter_ws_size, bwd_data_ws_size;

#if CUDNN_VERSION >= 3000
    if (exhaustive_search_) {
      CAFFE_VLOG(1) << "CUDNN Convolution bwd: doing exhaustive search.";
      // When we do an exhaustive search, we will ignore the workspace size
      // limit and simply go for the fastest algorithm. If you happen to run
      // out of memory later, you will be on your own...
      int returned_algo_count;
      // We clean up the current workspace memory so that the forward algorithm
      // is free to allocate memory.
      cudnn_ws_wrapper->Reset();
      // Actually run the search.
      cudnnConvolutionBwdFilterAlgoPerf_t filter_perf_stat;
      CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          bottom_desc_, top_desc_, conv_desc_, filter_desc_,
          1, &returned_algo_count, &filter_perf_stat));
      CAFFE_DCHECK_EQ(returned_algo_count, 1);
      bwd_filter_algo_ = filter_perf_stat.algo;
      cudnnConvolutionBwdDataAlgoPerf_t data_perf_stat;
      CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          filter_desc_, top_desc_, conv_desc_, bottom_desc_,
          1, &returned_algo_count, &data_perf_stat));
      CAFFE_DCHECK_EQ(returned_algo_count, 1);
      bwd_data_algo_ = data_perf_stat.algo;
    } else {
      // choose backward algorithm for filter
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          bottom_desc_, top_desc_, conv_desc_, filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_, &bwd_filter_algo_));
      // choose backward algo for data
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
          cudnn_wrapper_.cudnn_handle(),
          filter_desc_, top_desc_, conv_desc_, bottom_desc_,
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_, &bwd_data_algo_));
    }
    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_wrapper_.cudnn_handle(),
        bottom_desc_, top_desc_, conv_desc_, filter_desc_,
        bwd_filter_algo_, &bwd_filter_ws_size));
    // get workspace for backwards data algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_wrapper_.cudnn_handle(),
        filter_desc_, top_desc_, conv_desc_, bottom_desc_,
        bwd_data_algo_, &bwd_data_ws_size));


    cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);
    CAFFE_VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
#endif  // CUDNN_VERSION >= 3000
  }

  // Now, actually run the computation.
  const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
  const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
  CUDNN_CHECK(cudnnConvolutionBackwardBias(
      cudnn_wrapper_.cudnn_handle(), &kOne, top_desc_, dY.template data<T>(),
      &kZero, bias_desc_, dbias->template mutable_data<T>()));
#if CUDNN_VERSION >= 3000
  CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(
      cudnn_wrapper_.cudnn_handle(), &kOne, bottom_desc_, X.template data<T>(),
      top_desc_, dY.template data<T>(), conv_desc_, bwd_filter_algo_,
      cudnn_ws_wrapper->Get(cudnn_ws_nbytes_), cudnn_ws_nbytes_,
      &kZero, filter_desc_, dfilter->template mutable_data<T>()));
#else  // CUDNN_VERSION >= 3000
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      cudnn_wrapper_.cudnn_handle(), &kOne, bottom_desc_, X.template data<T>(),
      top_desc_, dY.template data<T>(), conv_desc_,
      &kZero, filter_desc_, dfilter->template mutable_data<T>()));
#endif  // CUDNN_VERSION >= 3000

  if (OutputSize() == 3) {
    // Compute the gradient w.r.t. the input.
    auto *dX = Output(INPUT_GRAD);
    dX->ReshapeLike(X);
#if CUDNN_VERSION >= 3000
    CUDNN_CHECK(cudnnConvolutionBackwardData_v3(
        cudnn_wrapper_.cudnn_handle(), &kOne, filter_desc_,
        filter.template data<T>(), top_desc_, dY.template data<T>(),
        conv_desc_, bwd_data_algo_,
        cudnn_ws_wrapper->Get(cudnn_ws_nbytes_), cudnn_ws_nbytes_,
        &kZero, bottom_desc_, dX->template mutable_data<T>()));
#else  // CUDNN_VERSION >= 3000
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_wrapper_.cudnn_handle(), &kOne, filter_desc_,
        filter.template data<T>(), top_desc_, dY.template data<T>(),
        conv_desc_, &kZero, bottom_desc_, dX->template mutable_data<T>()));
#endif  // CUDNN_VERSION >= 3000
  }
  return true;
}

REGISTER_CUDNN_OPERATOR(Conv, CudnnConvOp<float>);
REGISTER_CUDNN_OPERATOR(ConvGradient, CudnnConvGradientOp<float>);

#if CUDNN_VERSION >= 3000
REGISTER_CUDNN_OPERATOR(ConvFp16, CudnnConvOp<float16>);
REGISTER_CUDNN_OPERATOR(ConvFp16Gradient, CudnnConvGradientOp<float16>);

struct GetConvFp16Gradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    CAFFE_CHECK_EQ(def.input_size(), 3);
    return SingleGradientDef(
        "ConvFp16Gradient", "",
        vector<string>{I(def, 0), I(def, 1), GO(def, 0)},
        vector<string>{GI(def, 1), GI(def, 2), GI(def, 0)});
  }
};
REGISTER_GRADIENT(ConvFp16, GetConvFp16Gradient);
#endif  // CUDNN_VERSION >= 3000

}  // namespace caffe2
