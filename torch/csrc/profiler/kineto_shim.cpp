#include <torch/csrc/profiler/kineto_shim.h>

#include <type_traits>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <c10/util/Exception.h>

namespace torch {
namespace profiler {
namespace impl {
namespace kineto {

// Here lies pain and `#ifdef USE_KINETO`

#ifdef USE_KINETO
namespace {
const std::set<libkineto::ActivityType> cpuTypes{
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // CUDA_RUNTIME appears in both cpuTypes and cudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,
};
} // namespace
#endif // USE_KINETO

static_assert(
    std::is_pod<DeviceAndResource>::value,
    "Kineto specific details should be in `kineto_ids`.");

const DeviceAndResource kineto_ids() {
#ifdef USE_KINETO
  return {
      /*device=*/libkineto::processId(),
      /*resource=*/libkineto::systemThreadId()};
#else
  return {};
#endif // USE_KINETO
}

void addMetadata(
    const activity_t* activity,
    const std::string& key,
    const std::string& value) {
#ifdef USE_KINETO
  // ActivityTraceInterface returns const pointers, so we have to cast away the
  // constness to add metadata.
  const_cast<activity_t*>(activity)->addMetadata(key, value);
#endif // USE_KINETO
}

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
#ifdef USE_KINETO
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}
#else
{
}
#endif // USE_KINETO

TraceWrapper::~TraceWrapper() = default;

activity_t* TraceWrapper::addCPUActivity(
    const std::string& name,
    const libkineto::ActivityType type,
    const DeviceAndResource device_and_resource,
    const uint64_t correlation_id,
    const int64_t start_time,
    const int64_t end_time) {
#ifdef USE_KINETO
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);
  auto& act = libkineto::CpuTraceBuffer::toRef(cpu_trace_->activities.back());
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = correlation_id;
  act.startTime = start_time;
  if (type != libkineto::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return cpu_trace_->activities.back().get();
#else
  return nullptr;
#endif // USE_KINETO
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
#ifdef USE_KINETO
  cpu_trace_->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));
#endif // USE_KINETO
}

TraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return cpu_trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

ActivityTraceWrapper::ActivityTraceWrapper(
    std::unique_ptr<interface_trace_t>&& trace)
    : trace_(std::move(trace)) {}

ActivityTraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

void ActivityTraceWrapper::save(const std::string& path) {
#ifdef USE_KINETO
  TORCH_CHECK(!saved_, "Trace is already saved.");
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
#else
  TORCH_CHECK(
      false,
      "Saving a trace requires using torch.profiler with Kineto support (USE_KINETO=1)");
#endif // USE_KINETO
}

namespace {
// Handles processing of Experimental Config options for Kineto
class ExperimentalConfigWrapper {
 public:
  explicit ExperimentalConfigWrapper(
      const torch::profiler::impl::ExperimentalConfig& config)
      : config_(config) {}

  bool assertValid(const ActivitySet& activities) {
    // Kineto supports reading performance events per kernel/iteration
    // using CUPTI Range based profiler API. In this mode however we
    // do not trace CPU or GPU events.
    bool cupti_range_profiler = config_.profiler_metrics.size() > 0;
    if (cupti_range_profiler &&
        activities.count(torch::autograd::profiler::ActivityType::CPU)) {
      LOG(WARNING)
          << "Cannot run range profiler with CPU activities, please only"
          << " use CUDA activity type";
      return false;
    }
    return cupti_range_profiler;
  }

  void prepareTraceWithExperimentalOptions() {
#ifdef USE_KINETO
    std::set<libkineto::ActivityType> k_activities{
        libkineto::ActivityType::CUDA_PROFILER_RANGE};

    const size_t num_metrics = config_.profiler_metrics.size();
    std::stringstream configss;

    LOG(INFO) << "CUPTI profiler metrics size = " << num_metrics;

    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
             << "CUPTI_PROFILER_METRICS=";

    for (int i = 0; i < num_metrics; i++) {
      configss << config_.profiler_metrics[i];
      if (num_metrics > 1 && i < (num_metrics - 1)) {
        configss << ",";
      }
    }
    configss << "\nCUPTI_PROFILER_ENABLE_PER_KERNEL="
             << (config_.profiler_measure_per_kernel ? "true" : "false")
             << "\n";
    LOG(INFO) << "Generated config = " << configss.str();

    libkineto::api().activityProfiler().prepareTrace(
        k_activities, configss.str());
#endif // USE_KINETO
  }

 private:
  const torch::profiler::impl::ExperimentalConfig& config_;
};
} // namespace

void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config) {
#ifdef USE_KINETO
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(torch::autograd::profiler::ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::CUDA)) {
    k_activities.insert(cudaTypes.begin(), cudaTypes.end());
  }

  ExperimentalConfigWrapper configWrap(config);

  // Experimental Configuration options are present
  if (config && configWrap.assertValid(activities)) {
    configWrap.prepareTraceWithExperimentalOptions();
    return;
  }

  libkineto::api().activityProfiler().prepareTrace(k_activities);
#endif // USE_KINETO
}

void startTrace() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().startTrace();
#endif // USE_KINETO
}

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{
#ifdef USE_KINETO
      libkineto::api().activityProfiler().stopTrace()
#else
      std::make_unique<interface_trace_t>()
#endif // USE_KINETO
  };
}

void pushCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushCorrelationId(correlation_id);
#endif // USE_KINETO
}

void pushUserCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushUserCorrelationId(correlation_id);
#endif // USE_KINETO
}

void popCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
}

void popUserCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popUserCorrelationId();
#endif // USE_KINETO
}

void recordThreadInfo() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().recordThreadInfo();
#endif // USE_KINETO
}

} // namespace kineto
} // namespace impl
} // namespace profiler

namespace autograd {
namespace profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
  // fallthrough
  switch (activity_type) {
    case libkineto::ActivityType::GPU_MEMCPY:
    case libkineto::ActivityType::GPU_MEMSET:
    case libkineto::ActivityType::CONCURRENT_KERNEL:
    case libkineto::ActivityType::GPU_USER_ANNOTATION:
    case libkineto::ActivityType::CUDA_PROFILER_RANGE:
      return c10::DeviceType::CUDA;
    case libkineto::ActivityType::CPU_OP:
    case libkineto::ActivityType::USER_ANNOTATION:
    case libkineto::ActivityType::EXTERNAL_CORRELATION:
    case libkineto::ActivityType::CUDA_RUNTIME:
    case libkineto::ActivityType::CPU_INSTANT_EVENT:
    case libkineto::ActivityType::GLOW_RUNTIME:
    case libkineto::ActivityType::PYTHON_FUNCTION:
      return c10::DeviceType::CPU;
    default: {
      TORCH_WARN(
          "Unknown activity type (",
          (uint8_t)activity_type,
          "), assuming CPU device");
      return c10::DeviceType::CPU;
    }
  }
}

void addMetadataJson(const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
#else
  LOG(WARNING) << "Adding profiling metadata requires using "
               << "torch.profiler with Kineto support (USE_KINETO=1)";
#endif // USE_KINETO
}

void profilerStep() {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().step();
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping step() invocation";
  }
#endif // USE_KINETO
}

} // namespace profiler
} // namespace autograd
} // namespace torch
