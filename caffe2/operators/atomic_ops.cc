#include <mutex>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

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
    std::lock_guard<std::mutex> lg(*mutex);
    c->Resize();
    d->Resize();
    auto* aPtr = a.data<int32_t>();
    auto* bPtr = b.data<int32_t>();
    auto* cPtr = c->template mutable_data<int32_t>();
    auto* dPtr = d->template mutable_data<int32_t>();
    *dPtr = *aPtr;
    *cPtr = *aPtr + *bPtr;
    return true;
  }
};

class CreateAtomicBoolOp final : public Operator<CPUContext> {
 public:
  using Operator::Operator;

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<std::atomic<bool>>>(0) =
        std::unique_ptr<std::atomic<bool>>(new std::atomic<bool>(false));
    return true;
  }
};

class ConditionalSetAtomicBoolOp final : public Operator<CPUContext> {
 public:
  using Operator::Operator;

  bool RunOnDevice() override {
    auto& ptr =
        OperatorBase::Input<std::unique_ptr<std::atomic<bool>>>(ATOMIC_BOOL);
    if (Input(CONDITION).data<bool>()[0]) {
      ptr->store(true);
    }
    return true;
  }

 private:
  INPUT_TAGS(ATOMIC_BOOL, CONDITION);
};

class CheckAtomicBoolOp final : public Operator<CPUContext> {
 public:
  using Operator::Operator;

  bool RunOnDevice() override {
    auto& ptr = OperatorBase::Input<std::unique_ptr<std::atomic<bool>>>(0);
    Output(0)->Resize(1);
    *Output(0)->template mutable_data<bool>() = ptr->load();
    return true;
  }
};

REGISTER_CPU_OPERATOR(CreateMutex, CreateMutexOp);
REGISTER_CPU_OPERATOR(AtomicFetchAdd, AtomicFetchAddOp);

#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(CreateMutex, IDEEPFallbackOp<CreateMutexOp, SkipIndices<0>>);
#endif

REGISTER_CPU_OPERATOR(CreateAtomicBool, CreateAtomicBoolOp);
REGISTER_CPU_OPERATOR(ConditionalSetAtomicBool, ConditionalSetAtomicBoolOp);
REGISTER_CPU_OPERATOR(CheckAtomicBool, CheckAtomicBoolOp);

OPERATOR_SCHEMA(CreateMutex)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Creates an unlocked mutex and returns it in a unique_ptr blob.")
    .Output(0, "mutex_ptr", "Blob containing a std::unique_ptr<mutex>.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

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

OPERATOR_SCHEMA(CreateAtomicBool)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Create an unique_ptr blob to hold an atomic<bool>")
    .Output(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>");

OPERATOR_SCHEMA(ConditionalSetAtomicBool)
    .NumInputs(2)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Set an atomic<bool> to true if the given condition bool variable is true
    )DOC")
    .Input(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
    .Input(1, "condition", "Blob containing a bool");

OPERATOR_SCHEMA(CheckAtomicBool)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Copy the value of an atomic<bool> to a bool")
    .Input(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
    .Output(0, "value", "Copy of the value for the atomic<bool>");

SHOULD_NOT_DO_GRADIENT(CreateMutex);
SHOULD_NOT_DO_GRADIENT(AtomicFetchAdd);
SHOULD_NOT_DO_GRADIENT(CreateAtomicBool);
SHOULD_NOT_DO_GRADIENT(ConditionalSetAtomicBool);
SHOULD_NOT_DO_GRADIENT(CheckAtomicBool);
}
}
}
