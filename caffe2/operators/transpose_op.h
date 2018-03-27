#ifndef CAFFE2_OPERATORS_TRANSPOSE_H_
#define CAFFE2_OPERATORS_TRANSPOSE_H_
#define MAX_BLOB_NUM 1024

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class TransposeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;
  TransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
    // We will check the legality of axes_: it should be from 0 to axes_.size().
    std::vector<int> axes_sorted(axes_);
    std::sort(axes_sorted.begin(), axes_sorted.end());
    for (int i = 0; i < axes_sorted.size(); ++i) {
      if (axes_sorted[i] != i) {
        CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
      }
    }
  }
  ~TransposeOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int num_axes = X.ndim();
    const std::vector<int> x_dims(X.dims().cbegin(), X.dims().cend());
    std::vector<int> y_dims(num_axes);
    if (axes_.empty()) {
      axes_.resize(num_axes);
      for (int i = 0; i < num_axes; ++i) {
        axes_[i] = num_axes - 1 - i;
      }
      y_dims.assign(X.dims().rbegin(), X.dims().rend());
    } else {
      CAFFE_ENFORCE_EQ(X.ndim(), axes_.size());
      for (int i = 0; i < num_axes; ++i) {
        y_dims[i] = X.dim32(axes_[i]);
      }
    }
    Y->Resize(y_dims);
    SetDeviceTensor(x_dims, &x_dims_device_);
    SetDeviceTensor(y_dims, &y_dims_device_);
    SetDeviceTensor(axes_, &axes_device_);

    // Do the actual transpose, which is implemented in DoRunWithType().
    return DispatchHelper<TensorTypes<float, double, int, long>>::call(
        this, Input(0));
  }

 protected:
  void SetDeviceTensor(const std::vector<int>& data, Tensor<Context>* tensor) {
    tensor->Resize(data.size());
    context_.template Copy<int, CPUContext, Context>(
        data.size(), data.data(), tensor->template mutable_data<int>());
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    math::Transpose<T, Context>(
        axes_.size(),
        x_dims_device_.template data<int>(),
        y_dims_device_.template data<int>(),
        axes_device_.template data<int>(),
        X.size(),
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

  std::vector<int> axes_;

  Tensor<Context> x_dims_device_;
  Tensor<Context> y_dims_device_;
  Tensor<Context> axes_device_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TRANSPOSE_H_
