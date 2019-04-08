#include <chrono>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

class StatRegistryCreateOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit StatRegistryCreateOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<StatRegistry>>(0) =
        std::unique_ptr<StatRegistry>(new StatRegistry);
    return true;
  }
};

class StatRegistryExportOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit StatRegistryExportOp(Args&&... args)
      : Operator(std::forward<Args>(args)...),
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
    auto* pkeys = keys->template mutable_data<std::string>();
    auto* pvals = values->template mutable_data<int64_t>();
    auto* ptimestamps = timestamps->template mutable_data<int64_t>();
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
  template <class... Args>
  explicit StatRegistryUpdateOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    const auto& keys = Input(0);
    const auto& values = Input(1);
    auto registry = InputSize() == 3
        ? OperatorBase::Input<std::unique_ptr<StatRegistry>>(2).get()
        : &StatRegistry::get();
    CAFFE_ENFORCE_EQ(keys.numel(), values.numel());
    ExportedStatList data(keys.numel());
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
  explicit TimerBeginOp(const OperatorDef& operator_def, Workspace* ws)
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
  template <class... Args>
  explicit TimerEndOp(Args&&... args) : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    OperatorBase::Input<TimerInstance*>(0)->end();
    return true;
  }
};

struct TimerGetAndEndOp : public Operator<CPUContext> {
  template <class... Args>
  explicit TimerGetAndEndOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    int64_t nanos = OperatorBase::Input<TimerInstance*>(0)->get_ns();
    OperatorBase::Input<TimerInstance*>(0)->end();
    auto* res = Output(0);
    res->Resize(1);
    res->template mutable_data<int64_t>()[0] = nanos;
    return true;
  }
};

struct TimerGetOp : public Operator<CPUContext> {
  template <class... Args>
  explicit TimerGetOp(Args&&... args) : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    int64_t nanos = OperatorBase::Input<TimerInstance*>(0)->get_ns();
    auto* res = Output(0);
    res->Resize();
    res->template mutable_data<int64_t>()[0] = nanos;
    return true;
  }
};

REGISTER_CPU_OPERATOR(StatRegistryCreate, StatRegistryCreateOp);
REGISTER_CPU_OPERATOR(StatRegistryUpdate, StatRegistryUpdateOp);
REGISTER_CPU_OPERATOR(StatRegistryExport, StatRegistryExportOp);

REGISTER_CPU_OPERATOR(TimerBegin, TimerBeginOp);
REGISTER_CPU_OPERATOR(TimerEnd, TimerEndOp);
REGISTER_CPU_OPERATOR(TimerGetAndEnd, TimerGetAndEndOp);
REGISTER_CPU_OPERATOR(TimerGet, TimerGetOp);

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
Start a wallclock timer, returning a scalar tensor containing a pointer to it. The timer is stopped by calling **TimerEnd**.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc

    )DOC")
    .Arg("counter_name", "(*str*): name of the timer object; if not set use output name")
    .Output(0, "timer", "(*Tensor`<ptr>`*): pointer to a timer object");

OPERATOR_SCHEMA(TimerEnd)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Stop a timer started with **TimerBegin**. Publishes a CAFFE_EVENT.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc

    )DOC")
    .Input(0, "timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op");

OPERATOR_SCHEMA(TimerGetAndEnd)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Queries the current time of a timer in nanos, stops the timer publishing a CAFFE_EVENT.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

timerbegin_op = core.CreateOperator(
    "TimerBegin",
    [],
    ["timer"]
)

timerget_op = core.CreateOperator(
    "TimerGet",
    ["timer"],
    ["nanos"]
)

timerend_op = core.CreateOperator(
    "TimerEnd",
    ["timer"],
    []
)

timergetandend_op = core.CreateOperator(
    "TimerGetAndEnd",
    ["timer"],
    ["nanos"]
)

// Test TimerBegin/TimerGet/TimerEnd
workspace.RunOperatorOnce(timerbegin_op)
print("timer:", workspace.FetchBlob("timer"))
workspace.RunOperatorOnce(timerget_op)
print("nanos:", workspace.FetchBlob("nanos"))
workspace.RunOperatorOnce(timerend_op)


// Test TimerBegin/TimerGetAndEnd
workspace.RunOperatorOnce(timerbegin_op)
print("timer:", workspace.FetchBlob("timer"))
workspace.RunOperatorOnce(timergetandend_op)
print("nanos:", workspace.FetchBlob("nanos"))

```

**Result**

```

timer: b'timer, a C++ native class of type caffe2::TimerInstance*.'
nanos: 361140
timer: b'timer, a C++ native class of type caffe2::TimerInstance*.'
nanos: [252250]

```

</details>

      )DOC")
    .Input(0, "timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op")
    .Output(0, "nanos", "(*Tensor`<int64>`*): scalar tensor containing time in nanoseconds");

OPERATOR_SCHEMA(TimerGet)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Queries the current time of a timer object in nanoseconds.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc

    )DOC")
    .Input(0, "timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op")
    .Output(0, "nanos", "(*Tensor`<int64>`*): scalar containing time in nanoseconds");

CAFFE_KNOWN_TYPE(TimerInstance*);
CAFFE_KNOWN_TYPE(std::unique_ptr<caffe2::StatRegistry>);
} // namespace caffe2
