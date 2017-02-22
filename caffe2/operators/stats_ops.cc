#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

class StatRegistryCreateOp : public Operator<CPUContext> {
 public:
  StatRegistryCreateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<StatRegistry>>(0) =
        std::unique_ptr<StatRegistry>(new StatRegistry);
    return true;
  }
};

class StatRegistryExportOp : public Operator<CPUContext> {
 public:
  StatRegistryExportOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        reset_(GetSingleArgument<bool>("reset", true)) {}

  bool RunOnDevice() override {
    auto registry = InputSize() > 0
        ? OperatorBase::Input<std::unique_ptr<StatRegistry>>(0).get()
        : &StatRegistry::get();
    auto* keys = Output(0);
    auto* values = Output(1);
    auto* timestamps = Output(2);
    auto data = registry->publish(reset_);
    keys->Resize(data.size());
    values->Resize(data.size());
    timestamps->Resize(data.size());
    auto* pkeys = keys->mutable_data<std::string>();
    auto* pvals = values->mutable_data<int64_t>();
    auto* ptimestamps = timestamps->mutable_data<int64_t>();
    int i = 0;
    for (const auto& stat : data) {
      pkeys[i] = std::move(stat.key);
      pvals[i] = stat.value;
      ptimestamps[i] =
          std::chrono::nanoseconds(stat.ts.time_since_epoch()).count();
      ++i;
    }
    return true;
  }

 private:
  bool reset_;
};

class StatRegistryUpdateOp : public Operator<CPUContext> {
 public:
  StatRegistryUpdateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& keys = Input(0);
    const auto& values = Input(1);
    auto registry = InputSize() == 3
        ? OperatorBase::Input<std::unique_ptr<StatRegistry>>(2).get()
        : &StatRegistry::get();
    CAFFE_ENFORCE_EQ(keys.size(), values.size());
    ExportedStatList data(keys.size());
    auto* pkeys = keys.data<std::string>();
    auto* pvals = values.data<int64_t>();
    int i = 0;
    for (auto& stat : data) {
      stat.key = pkeys[i];
      stat.value = pvals[i];
      ++i;
    }
    registry->update(data);
    return true;
  }
};

REGISTER_CPU_OPERATOR(StatRegistryCreate, StatRegistryCreateOp);
REGISTER_CPU_OPERATOR(StatRegistryUpdate, StatRegistryUpdateOp);
REGISTER_CPU_OPERATOR(StatRegistryExport, StatRegistryExportOp);

OPERATOR_SCHEMA(StatRegistryCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Create a StatRegistry object that will contain a map of performance counters
keyed by name. A StatRegistry is used to gather and retrieve performance
counts throuhgout the caffe2 codebase.
)DOC")
    .Output(0, "handle", "A Blob pointing to the newly created StatRegistry.");

OPERATOR_SCHEMA(StatRegistryUpdate)
    .NumInputs(2, 3)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Update the given StatRegistry, or the global StatRegistry,
with the values of counters for the given keys.
)DOC")
    .Input(0, "keys", "1D string tensor with the key names to update.")
    .Input(1, "values", "1D int64 tensor with the values to update.")
    .Input(
        2,
        "handle",
        "If provided, update the given StatRegistry. "
        "Otherwise, update the global singleton.");

OPERATOR_SCHEMA(StatRegistryExport)
    .NumInputs(0, 1)
    .NumOutputs(3)
    .Input(
        0,
        "handle",
        "If provided, export values from given StatRegistry."
        "Otherwise, export values from the global singleton StatRegistry.")
    .Output(0, "keys", "1D string tensor with exported key names")
    .Output(1, "values", "1D int64 tensor with exported values")
    .Output(2, "timestamps", "The unix timestamp at counter retrieval.")
    .Arg(
        "reset",
        "(default true) Whether to atomically reset the counters afterwards.");
}

CAFFE_KNOWN_TYPE(std::unique_ptr<caffe2::StatRegistry>);
} // namespace caffe2
