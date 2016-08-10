#ifndef CAFFE2_SGD_ITER_OP_H_
#define CAFFE2_SGD_ITER_OP_H_
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// IterOp runs an iteration counter. I cannot think of a case where we would
// need to access the iter variable on device, so this will always produce a
// tensor on the CPU side. If the blob already exists and is a tensor<int>
// object, we will simply increment it (this emulates the case when we want to
// resume training). Otherwise we will have the iter starting with 0.
template <class Context>
class IterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  IterOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    if (InputSize() == 0) {
      LOG(ERROR) << "You are using an old definition of IterOp that will "
                         "be deprecated soon. More specifically, IterOp now "
                         "requires an explicit in-place input and output.";
      if (!OperatorBase::OutputIsType<TensorCPU>(0)) {
        // This is the first run; set the iter to start with 0.
        auto* output = OperatorBase::Output<TensorCPU>(0);
        VLOG(1) << "Initializing iter counter.";
        output->Resize(1);
        output->template mutable_data<int>()[0] = 0;
      }
    }
    auto* output = OperatorBase::Output<TensorCPU>(0);
    CHECK_EQ(output->size(), 1)
        << "The output of IterOp exists, but not of the right size.";
    int* iter = output->template mutable_data<int>();
    CAFFE_ENFORCE(*iter >= 0, "Previous iteration number is negative.");
    CAFFE_ENFORCE(*iter < INT_MAX, "Overflow will happen!");
    (*iter)++;
    return true;
  }

 private:
};

} // namespace caffe2

#endif // CAFFE2_SGD_ITER_OP_H_
