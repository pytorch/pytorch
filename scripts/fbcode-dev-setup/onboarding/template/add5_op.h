#ifndef CAFFE_OPERATORS_ADD5_OP_H_
#define CAFFE_OPERATORS_ADD5_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class Add5Op final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Add5Op(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    // In this function, we usually store the argument as a member of the
    // object. No need to do anything in this simple example.
  }

  bool RunOnDevice() override {
    // Instantiate the template for int/int64_t/float/double tensors.
    // For details, check:
    // https://github.com/pytorch/pytorch/blob/master/caffe2/core/operator.h
    return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(DATA);

 private:
  // Object fields are put here.
};

template <class Context>
class Add5GradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Add5GradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    // In this function, we usually store the argument as a member of the
    // object. No need to do anything in this simple example.
  }

  bool RunOnDevice() override {
    // Instantiate the template for int/int64_t/float/double tensors.
    // For details, check:
    // https://github.com/pytorch/pytorch/blob/master/caffe2/core/operator.h
    return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(DATA);

 private:
  // Object fields are put here.
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ADD5_OP_H_
