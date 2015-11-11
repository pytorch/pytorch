#ifndef CAFFE2_OPERATORS_RELU_OP_H_
#define CAFFE2_OPERATORS_RELU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
class ReluOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(ReluOp);
};

template <typename T, class Context>
class ReluGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: Y, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  IN_PLACE_ALLOWED({1, 0});
  DISABLE_COPY_AND_ASSIGN(ReluGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_RELU_OP_H_
