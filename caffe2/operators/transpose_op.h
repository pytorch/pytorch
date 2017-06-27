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
  bool DoRunWithType();

  std::vector<int> axes_;
  std::vector<TIndex> new_dims_;
  // buffer_ is used in TransposeOp<CUDAContext> so we can obtain a consistent
  // buffer on the GPU. It is not used in the CPUContext implementation.
  Tensor<Context> buffer_;
  TensorCPU buffer_cpu_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TRANSPOSE_H_
