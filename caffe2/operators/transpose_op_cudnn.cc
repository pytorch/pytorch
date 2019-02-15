#include "caffe2/operators/transpose_op.h"

#include <algorithm>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

#define MAX_DIMS 8

class CuDNNTransposeOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  USE_DISPATCH_HELPER;

  CuDNNTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
    // We will check the legality of axes_: it should be from 0 to axes_.size().
    std::vector<int> axes_sorted(axes_);
    std::sort(axes_sorted.begin(), axes_sorted.end());
    for (int i = 0; i < axes_sorted.size(); ++i) {
      if (axes_sorted[i] != i) {
        CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
      }
    }

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&xDesc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&yDesc_));
  }

  ~CuDNNTransposeOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(xDesc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(yDesc_));
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.dim();
    X_dims_.assign(X.sizes().cbegin(), X.sizes().cend());
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.rbegin(), axes_.rend(), 0);
    } else {
      CAFFE_ENFORCE_EQ(X.dim(), axes_.size());
    }
    std::vector<int> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims_[axes_[i]];
    }
    Y->Resize(Y_dims);
    // Do the actual transpose, which is implemented in DoRunWithType().
#if CUDNN_VERSION_MIN(6, 0, 0)
    return DispatchHelper<TensorTypes<float, int>>::call(this, Input(0));
#else
    // CUDNN 5.1 does not have int support yet.
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
#endif
  }

 protected:
  template <typename T>
  bool DoRunWithType() {
    const auto& input = Input(0);
    auto* output = Output(0);
    int ndim = input.dim();

    if (ndim == 0) {
      return true;
    }
    if (ndim == 1) {
      output->CopyFrom(input);
      return true;
    }

    cudnnDataType_t typedesc = cudnnTypeWrapper<T>::type;
#if CUDNN_VERSION_MIN(6, 0, 0)
    if (typedesc == CUDNN_DATA_INT32) {
      // CUDNN Transpose only support float for now
      math::Transpose<int, CUDAContext>(
          X_dims_.size(),
          X_dims_.data(),
          axes_.data(),
          input.template data<int>(),
          output->template mutable_data<int>(),
          &context_);
      return true;
    }
#endif

    CAFFE_ENFORCE(ndim < MAX_DIMS, "Input ndim exceeds compile time max.");

    stride_y[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      stride_y[i] = stride_y[i + 1] * output->dim32(i + 1);
    }

    CHECK(axes_.size() >= ndim);

    stride_x[ndim] = 1;
    for (int i = 0; i < ndim; i++) {
      stride_x[i] = 1;
      for (int j = axes_[i] + 1; j < ndim; j++) {
        stride_x[i] *= input.dim32(j);
      }
      dim_y_int[i] = output->dim32(i);
    }

    // CuDNN requires at least 3-dim tensors
    for (int i = ndim; i < MAX_DIMS; i++) {
      stride_x[i] = 1;
      stride_y[i] = 1;
      dim_y_int[i] = 1;
    }

    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        xDesc_, typedesc, ndim < 4 ? 4 : ndim, dim_y_int, stride_x));

    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        yDesc_, typedesc, ndim < 4 ? 4 : ndim, dim_y_int, stride_y));

    CUDNN_ENFORCE(cudnnTransformTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        xDesc_,
        static_cast<const void*>(input.template data<T>()),
        cudnnTypeWrapper<T>::kZero(),
        yDesc_,
        static_cast<void*>(output->template mutable_data<T>())));
    return true;
  }

  int stride_x[MAX_DIMS];
  int stride_y[MAX_DIMS];
  int dim_y_int[MAX_DIMS];

  cudnnTensorDescriptor_t xDesc_;
  cudnnTensorDescriptor_t yDesc_;
  CuDNNWrapper cudnn_wrapper_;

  std::vector<int> axes_;
  std::vector<int> X_dims_;
};

REGISTER_CUDNN_OPERATOR(Transpose, CuDNNTransposeOp);

} // namespace caffe2
