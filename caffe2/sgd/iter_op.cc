#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// IterOp runs an iteration counter. I cannot think of a case where we would
// need to access the iter variable on device, so this will always produce a
// tensor on the CPU side. If the blob already exists and is a tensor<int>
// object, we will simply increment it (this emulates the case when we want to
// resume training). Otherwise we will have the iter starting with 0.
class IterOp final : public OperatorBase {
 public:
  IterOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}

  bool Run() override {
    if (InputSize() == 0) {
      CAFFE_LOG_ERROR << "You are using an old definition of IterOp that will "
                         "be deprecated soon. More specifically, IterOp now "
                         "requires an explicit in-place input and output.";
      if (!OperatorBase::OutputIsType<TensorCPU>(0)) {
        // This is the first run; set the iter to start with 0.
        auto* output = OperatorBase::Output<TensorCPU>(0);
        CAFFE_VLOG(1) << "Initializing iter counter.";
        output->Reshape(vector<TIndex>{1});
        output->mutable_data<int>()[0] = 0;
      }
    }
    auto* output = OperatorBase::Output<TensorCPU>(0);
    CAFFE_CHECK_EQ(output->size(), 1)
        << "The output of IterOp exists, but not of the right size.";
    int* iter = output->mutable_data<int>();
    CAFFE_CHECK_GE(*iter, 0) << "Previous iteration number is negative.";
    CAFFE_CHECK_LT(*iter, INT_MAX) << "Overflow will happen!";
    (*iter)++;
    return true;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(IterOp);
};

namespace {
REGISTER_CPU_OPERATOR(Iter, IterOp);
REGISTER_CUDA_OPERATOR(Iter, IterOp);

OPERATOR_SCHEMA(Iter).NumInputs(0, 1).NumOutputs(1).EnforceInplace({{0, 0}});

NO_GRADIENT(Iter);
}
}  // namespace caffe2
