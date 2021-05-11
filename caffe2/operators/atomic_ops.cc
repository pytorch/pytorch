#include <mutex>
#include <thread>
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
  template <class... Args>
  explicit CreateMutexOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<std::mutex>>(0) =
        // NOLINTNEXTLINE(modernize-make-unique)
        std::unique_ptr<std::mutex>(new std::mutex);
    return true;
  }
};

template <typename IntType>
class AtomicFetchAddOp final : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit AtomicFetchAddOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& mutex = OperatorBase::Input<std::unique_ptr<std::mutex>>(0);
    std::lock_guard<std::mutex> lg(*mutex);
    auto& a = Input(1);
    auto& b = Input(2);
    auto* c = Output(0);
    auto* d = Output(1);
    c->Resize();
    d->Resize();
    auto* aPtr = a.template data<IntType>();
    auto* bPtr = b.template data<IntType>();
    auto* cPtr = c->template mutable_data<IntType>();
    auto* dPtr = d->template mutable_data<IntType>();
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
        // NOLINTNEXTLINE(modernize-make-unique)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CreateMutex, CreateMutexOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(AtomicFetchAdd, AtomicFetchAddOp<int32_t>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(AtomicFetchAdd64, AtomicFetchAddOp<int64_t>);

#ifdef CAFFE2_USE_MKLDNN
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(
    CreateMutex,
    IDEEPFallbackOp<CreateMutexOp, SkipIndices<0>>);
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CreateAtomicBool, CreateAtomicBoolOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ConditionalSetAtomicBool, ConditionalSetAtomicBoolOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CheckAtomicBool, CheckAtomicBoolOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CreateMutex)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Creates an unlocked mutex and returns it in a unique_ptr blob.")
    .Output(0, "mutex_ptr", "Blob containing a std::unique_ptr<mutex>.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(AtomicFetchAdd64)
    .NumInputs(3)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Like, AtomicFetchAdd but with int64_t scalar tensors,
performs an atomic fetch add
by mutating the first argument and adding it to the second input
argument. Returns the updated integer and the value prior to the update.
)DOC")
    .Input(0, "mutex_ptr", "Blob containing to a unique_ptr<mutex>")
    .Input(1, "mut_value", "Value to be mutated after the sum.")
    .Input(2, "increment", "Value to add to the first operand.")
    .Output(0, "mut_value", "Mutated value after sum. Usually same as input 1.")
    .Output(1, "fetched_value", "Value of the first operand before sum.")
    .AllowInplace({{1, 0}});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CreateAtomicBool)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Create an unique_ptr blob to hold an atomic<bool>")
    .Output(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ConditionalSetAtomicBool)
    .NumInputs(2)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Set an atomic<bool> to true if the given condition bool variable is true
    )DOC")
    .Input(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
    .Input(1, "condition", "Blob containing a bool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CheckAtomicBool)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Copy the value of an atomic<bool> to a bool")
    .Input(0, "atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
    .Output(0, "value", "Copy of the value for the atomic<bool>");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CreateMutex);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(AtomicFetchAdd);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CreateAtomicBool);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(ConditionalSetAtomicBool);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CheckAtomicBool);
} // namespace
} // namespace fb
} // namespace caffe2
