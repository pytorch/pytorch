#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/transpose_op.h"

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

  ~CuDNNTransposeOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(xDesc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(yDesc_));
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    new_dims_.resize(X.ndim());
    if (axes_.size() == 0) {
      axes_.resize(X.ndim());
      for (int i = 0; i < axes_.size(); ++i) {
        axes_[i] = axes_.size() - 1 - i;
      }
      new_dims_.assign(X.dims().rbegin(), X.dims().rend());
    } else {
      CAFFE_ENFORCE_EQ(X.ndim(), axes_.size());
      for (int i = 0; i < new_dims_.size(); ++i) {
        new_dims_[i] = X.dim(axes_[i]);
      }
    }
    Y->Resize(new_dims_);
    // Do the actual transpose, which is implemented in DoRunWithType().
    return DispatchHelper<TensorTypes<float, double, int, long>>::call(
        this, Input(0));
  }

 protected:
  template <typename T>
  bool DoRunWithType() {
    const auto& input = Input(0);
    auto* output = Output(0);
    int ndim = input.ndim();

    if (ndim == 0) {
      return true;
    }
    if (ndim == 1) {
      output->CopyFrom(input);
      return true;
    }

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

    // Hack, since CUDNN only supports float types
    // TODO: half-float support
    CHECK(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double));
    cudnnDataType_t typedesc = sizeof(T) == sizeof(float)
        ? cudnnTypeWrapper<float>::type
        : cudnnTypeWrapper<double>::type;

    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        xDesc_, typedesc, ndim < 4 ? 4 : ndim, dim_y_int, stride_x));

    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        yDesc_, typedesc, ndim < 4 ? 4 : ndim, dim_y_int, stride_y));

    CUDNN_ENFORCE(cudnnTransformTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        sizeof(T) == sizeof(float) ? static_cast<const void*>(&alpha_)
                                   : static_cast<const void*>(&alphad_),
        xDesc_,
        static_cast<const void*>(input.template data<T>()),
        sizeof(T) == sizeof(float) ? static_cast<const void*>(&beta_)
                                   : static_cast<const void*>(&betad_),
        yDesc_,
        static_cast<void*>(output->template mutable_data<T>())));
    return true;
  }

  const float alpha_ = 1.0;
  const float beta_ = 0.0;
  const double alphad_ = 1.0;
  const double betad_ = 0.0;
  int stride_x[MAX_DIMS];
  int stride_y[MAX_DIMS];
  int dim_y_int[MAX_DIMS];

  cudnnTensorDescriptor_t xDesc_;
  cudnnTensorDescriptor_t yDesc_;
  CuDNNWrapper cudnn_wrapper_;
  std::vector<int> axes_;
  std::vector<TIndex> new_dims_;
};

REGISTER_CUDNN_OPERATOR(Transpose, CuDNNTransposeOp);

} // namespace caffe2
