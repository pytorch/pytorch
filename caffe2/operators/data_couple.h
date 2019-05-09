#ifndef CAFFE2_OPERATORS_NO_OP_OPTIMIZER_OP_H_
#define CAFFE2_OPERATORS_NO_OP_OPTIMIZER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class DataCoupleOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(DataCoupleOp)

  bool RunOnDevice() override {
    // Actually does nothing...
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NO_OP_OPTIMIZER_OP_H_
