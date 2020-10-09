#ifndef CAFFE2_OPERATORS_LOGIT_OP_H_
#define CAFFE2_OPERATORS_LOGIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_ops.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(Logit)

namespace caffe2 {

template <class Context>
struct LogitFunctor {
  explicit LogitFunctor(OperatorBase& op)
      : eps_(op.GetSingleArgument<float>("eps", 1e-6f)) {
    CAFFE_ENFORCE_GT(eps_, 0.0);
    CAFFE_ENFORCE_LT(eps_, 0.5);
  }

  template <typename T>
  bool operator()(const int size, const T* X, T* Y, Context* context) const;

  const float eps_;
};

template <typename T, class Context>
class LogitGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit LogitGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        eps_(this->template GetSingleArgument<float>("eps", 1e-6f)) {}
  ~LogitGradientOp() {}

  bool RunOnDevice() override;

 protected:
  float eps_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOGIT_OP_H_
