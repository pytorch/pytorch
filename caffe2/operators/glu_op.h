#ifndef CAFFE2_OPERATOR_GLU_OP_H_
#define CAFFE2_OPERATOR_GLU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
template <typename T, class Context>
class GluOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit GluOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        dim_(this->template GetSingleArgument<int>("dim", -1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() {
    auto& X = Input(0);

    vector<int64_t> Yshape;
    Yshape.insert(Yshape.end(), X.sizes().begin(), X.sizes().end());
    const int split_index = dim_ == -1 ? Yshape.size() - 1 : dim_;
    CAFFE_ENFORCE(
        Yshape[split_index] % 2 == 0,
        "Split dimension ",
        Yshape[split_index],
        " should be divided by two");
    const int split_dim_size = Yshape[split_index] / 2;
    const int M = X.size_to_dim(split_index);
    const int N = X.size_from_dim(split_index + 1);
    Yshape[split_index] = split_dim_size;
    auto* Y = Output(0, Yshape, at::dtype<T>());
    ComputeGlu(
        M,
        split_dim_size,
        N,
        X.template data<T>(),
        Y->template mutable_data<T>());
    return true;
  }

 protected:
  void ComputeGlu(
      const int M,
      const int split_dim_size,
      const int N,
      const T* X,
      T* output);

 private:
  const int dim_;
};
} // namespace caffe2

#endif // CAFFE2_OPERATOR_GLU_OP_H_
