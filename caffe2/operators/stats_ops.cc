#include <chrono>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

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

class TimerInstance {
 public:
  explicit TimerInstance(const std::string& name)
      : running_(false), stat_(name) {}

  void begin() {
    CAFFE_ENFORCE(!running_, "Called TimerBegin on an already running timer.");
    running_ = true;
    start_ = std::chrono::high_resolution_clock::now();
  }

  void end() {
    CAFFE_ENFORCE(running_, "Called TimerEnd on a stopped timer.");
    using namespace std::chrono;
    auto duration = high_resolution_clock::now() - start_;
    auto nanos = duration_cast<nanoseconds>(duration).count();
    CAFFE_EVENT(stat_, time_ns, nanos);
    running_ = false;
  }

  int64_t get_ns() {
    CAFFE_ENFORCE(running_, "Called TimerGet on a stopped timer.");
    using namespace std::chrono;
    auto duration = high_resolution_clock::now() - start_;
    auto nanos = duration_cast<nanoseconds>(duration).count();
    return nanos;
  }

 private:
  bool running_;
  std::chrono::high_resolution_clock::time_point start_;

  struct TimerStat {
    CAFFE_STAT_CTOR(TimerStat);
    CAFFE_AVG_EXPORTED_STAT(time_ns);
  } stat_;
};

struct TimerBeginOp : public Operator<CPUContext> {
  TimerBeginOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        given_name_(GetSingleArgument<std::string>(
            "counter_name",
            operator_def.output().Get(0))),
        timer_([this]() { return given_name_; }()) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<TimerInstance*>(0) = &timer_;
    timer_.begin();
    return true;
  }

 private:
  const std::string given_name_;
  TimerInstance timer_;
};

struct TimerEndOp : public Operator<CPUContext> {
  TimerEndOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    OperatorBase::Input<TimerInstance*>(0)->end();
    return true;
  }
};

struct TimerGetAndEndOp : public Operator<CPUContext> {
  TimerGetAndEndOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    int64_t nanos = OperatorBase::Input<TimerInstance*>(0)->get_ns();
    OperatorBase::Input<TimerInstance*>(0)->end();
    auto* res = OperatorBase::Output<TensorCPU>(0);
    res->Resize(1);
    res->template mutable_data<int64_t>()[0] = nanos;
    return true;
  }
};

struct TimerGetOp : public Operator<CPUContext> {
  TimerGetOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    int64_t nanos = OperatorBase::Input<TimerInstance*>(0)->get_ns();
    auto* res = OperatorBase::Output<TensorCPU>(0);
    res->Resize();
    res->template mutable_data<int64_t>()[0] = nanos;
    return true;
  }
};

struct CpuUtilizationReportOp : public Operator<CPUContext> {
  CpuUtilizationReportOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        statsName_(GetSingleArgument<std::string>("stats_name", "utilization")),
        stat_([this]() { return statsName_; }()) {}

  bool RunOnDevice() override {
    float utilization = Input(0).template data<float>()[0];
    // Utilization is a float value, but CAFFE_EVENT only keeps int64_t values.
    // We will keep 100x of the received utilization to maintain accuracy.
    CAFFE_EVENT(stat_, cpu_utilization, (int)(utilization * 100));
    return true;
  }

 private:
  std::string statsName_;
  struct CpuStats {
    CAFFE_STAT_CTOR(CpuStats);
    CAFFE_EXPORTED_STAT(cpu_utilization);
  } stat_;
};

REGISTER_CPU_OPERATOR(StatRegistryCreate, StatRegistryCreateOp);
REGISTER_CPU_OPERATOR(StatRegistryUpdate, StatRegistryUpdateOp);
REGISTER_CPU_OPERATOR(StatRegistryExport, StatRegistryExportOp);

REGISTER_CPU_OPERATOR(TimerBegin, TimerBeginOp);
REGISTER_CPU_OPERATOR(TimerEnd, TimerEndOp);
REGISTER_CPU_OPERATOR(TimerGetAndEnd, TimerGetAndEndOp);
REGISTER_CPU_OPERATOR(TimerGet, TimerGetOp);
REGISTER_CPU_OPERATOR(CpuUtilizationReport, CpuUtilizationReportOp);

OPERATOR_SCHEMA(StatRegistryCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Create a StatRegistry object that will contain a map of performance counters
keyed by name. A StatRegistry is used to gather and retrieve performance
counts throughout the caffe2 codebase.
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

OPERATOR_SCHEMA(TimerBegin)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Start a wallclock timer, returning a pointer to it.
The timer is stopped by calling TimerEnd)DOC")
    .Arg("counter_name", "Name of the timer. If not provided, use output name.")
    .Output(0, "timer", "Pointer to timer, to be passed to TimerEnd.");

OPERATOR_SCHEMA(TimerEnd)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc("Stop a timer started with TimerBegin, publishing a CAFFE_EVENT")
    .Input(0, "timer", "Pointer to timer, obtained from TimerBegin.");

OPERATOR_SCHEMA(TimerGetAndEnd)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Queries the current time of a timer in nanos, stops the timer
            publishing a CAFFE_EVENT)DOC")
    .Input(0, "timer", "Pointer to timer, obtained from TimerBegin.")
    .Output(0, "nanos", "nanoseconds in int64");

OPERATOR_SCHEMA(TimerGet)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Queries the current time of a timer in nanos)DOC")
    .Input(0, "timer", "Pointer to timer, obtained from TimerBegin.")
    .Output(0, "nanos", "nanoseconds in int64");

OPERATOR_SCHEMA(CpuUtilizationReport)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(Report the delta in max CPU utilization observed so far in the
            plan)DOC")
    .Input(
        0,
        "utilization",
        "Delta in max CPU utilization observed, in percentage as a float value")
    .Arg("stats_name", "String name of the stat entry holding CPU utilization");

CAFFE_KNOWN_TYPE(TimerInstance*);
CAFFE_KNOWN_TYPE(std::unique_ptr<caffe2::StatRegistry>);
} // namespace caffe2
