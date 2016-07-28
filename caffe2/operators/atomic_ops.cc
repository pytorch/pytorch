#include <mutex>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace fb {
namespace {

class CreateMutexOp final : public Operator<CPUContext> {
 public:
  CreateMutexOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<std::mutex>>(0) =
        std::unique_ptr<std::mutex>(new std::mutex);
    return true;
  }
};

class AtomicFetchAddOp final : public Operator<CPUContext> {
 public:
  AtomicFetchAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& mutex = OperatorBase::Input<std::unique_ptr<std::mutex>>(0);
    auto& a = Input(1);
    auto& b = Input(2);
    auto* c = Output(0);
    auto* d = Output(1);
    c->Resize(std::vector<TIndex>());
    d->Resize(std::vector<TIndex>());
    auto* aPtr = a.data<int32_t>();
    auto* bPtr = b.data<int32_t>();
    auto* cPtr = c->mutable_data<int32_t>();
    auto* dPtr = d->mutable_data<int32_t>();
    std::lock_guard<std::mutex> lg(*mutex);
    *dPtr = *aPtr;
    *cPtr = *aPtr + *bPtr;
    return true;
  }
};

REGISTER_CPU_OPERATOR(CreateMutex, CreateMutexOp);
REGISTER_CPU_OPERATOR(AtomicFetchAdd, AtomicFetchAddOp);

OPERATOR_SCHEMA(CreateMutex)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Creates an unlocked mutex and returns it in a unique_ptr blob.")
    .Output(0, "mutex_ptr", "Blob containing a std::unique_ptr<mutex>.");

OPERATOR_SCHEMA(AtomicFetchAdd)
    .NumInputs(3)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given a mutex and two int32 scalar tensors, performs an atomic fetch add
by mutating the first argument and adding it to the second input
argument. Returns the updated integer and the value prior to the update.
)DOC")
    .Input(0, "mutex_ptr", "Blob containing to a unique_ptr<mutex>")
    .Input(1, "mut_value", "Value to be mutated after the sum.")
    .Input(2, "increment", "Value to add to the first operand.")
    .Output(0, "mut_value", "Mutated value after sum. Usually same as input 1.")
    .Output(1, "fetched_value", "Value of the first operand before sum.")
    .AllowInplace({{1, 0}});

SHOULD_NOT_DO_GRADIENT(CreateMutex);
SHOULD_NOT_DO_GRADIENT(AtomicFetchAdd);
}
}
}
