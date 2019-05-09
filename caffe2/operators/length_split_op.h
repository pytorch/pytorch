#ifndef CAFFE2_OPERATORS_LENGTH_SPLIT_OP_H_
#define CAFFE2_OPERATORS_LENGTH_SPLIT_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class LengthsSplitOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit LengthsSplitOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        n_split_(OperatorBase::GetSingleArgument<int32_t>("n_split", 0)) {
    if (InputSize() == 1) {
      // If not specified, then must have this argument
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("n_split"),
          "Argument `n_split` is missing and was not specified as input.");
      CAFFE_ENFORCE(
          n_split_ > 0,
          "`n_split` must contain a positive value for defined behavior.");
    }
  }
  ~LengthsSplitOp() {}

  bool RunOnDevice() override {
    const auto& L = Input(0);
    CAFFE_ENFORCE_EQ(L.dim(), 1, "Input `LENGTHS` should be a 1D vector.");

    if (InputSize() > 1) {
      // We potentially have n_split specified as inputs as well
      CAFFE_ENFORCE(
          Input(1).dim() == 1 && Input(1).numel() == 1,
          "Input `n_split` should be a vector of size 1.");

      const auto& input1 = Input(1);
      context_.template CopyItems<Context, CPUContext>(
          input1.dtype(), 1, input1.raw_data(), &n_split_);
    }

    CAFFE_ENFORCE(
        n_split_ > 0,
        "`n_split` must contain a positive value for defined behavior.");
    const auto M = L.numel();

    auto* Y = Output(0, {M * n_split_}, at::dtype<int32_t>());

    const int32_t* Ldata = L.template data<int32_t>();
    int32_t* Ydata = Y->template mutable_data<int32_t>();

    for (int i = 0; i < M; i++) {
      int32_t mod = Ldata[i] % n_split_;
      int32_t res =
          mod != 0 ? math::DivUp(Ldata[i], n_split_) : Ldata[i] / n_split_ + 1;
      for (int j = 0; j < n_split_; j++) {
        Ydata[(i * n_split_) + j] = mod-- > 0 ? res : res - 1;
      }
    }
    return true;
  }

 private:
  int32_t n_split_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTH_SPLIT_OP_H_
