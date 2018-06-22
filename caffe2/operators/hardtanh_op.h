#ifndef CAFFE2_OPERATORS_HARDTANH_OP_H_
#define CAFFE2_OPERATORS_HARDTANH_OP_H_

#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class HardtanhOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  HardtanhOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    min_val_ = OperatorBase::GetSingleArgument<T>(
        "min_val", -1.0f);
    max_val_ = OperatorBase::GetSingleArgument<T>(
        "max_val", 1.0f);
    CAFFE_ENFORCE_GT(max_val_, min_val_);
  }

  bool RunOnDevice() override;

 protected:
  const T min_val_;
  const T max_val_;
};

template <typename T, class Context>
class HardtanhGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  HardtanhGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(T, "min_val", min_val_, -1.0f),
        OP_SINGLE_ARG(T, "max_val", max_val_, 1.0f) {
    CAFFE_ENFORCE_GT(max_val_, min_val_);
  }

  bool RunOnDevice() override;

 protected:
  T min_val_;
  T max_val_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_HARDTANH_OP_H_
