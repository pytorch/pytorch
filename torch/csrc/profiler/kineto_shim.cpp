#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <type_traits>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace torch {

namespace profiler::impl::kineto {

// Here lies pain and `#ifdef USE_KINETO`

#ifdef USE_KINETO
namespace {
const std::set<libkineto::ActivityType> kCpuTypes{
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::XPU_RUNTIME,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::PYTHON_FUNCTION,
    libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto::ActivityType::PRIVATEUSE1_DRIVER,
};

const std::set<libkineto::ActivityType> kCudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::GPU_USER_ANNOTATION,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // CUDA_RUNTIME appears in both kCpuTypes and kCudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::OVERHEAD,
};
const std::set<libkineto::ActivityType> kXpuTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // XPU_RUNTIME appears in both kCpuTypes and kXpuTypes.
    libkineto::ActivityType::XPU_RUNTIME,
};
const std::set<libkineto::ActivityType> kMtiaTypes = {
    libkineto::ActivityType::MTIA_CCP_EVENTS,
    libkineto::ActivityType::MTIA_RUNTIME,
    libkineto::ActivityType::MTIA_WORKLOADD,
};
const std::set<libkineto::ActivityType> hpuTypes = {
    libkineto::ActivityType::HPU_OP,
};
const std::set<libkineto::ActivityType> kPrivateUse1Types = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::GPU_USER_ANNOTATION,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // PRIVATEUSE1_RUNTIME appears in both kCpuTypes and kPrivateUse1Types.
    libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto::ActivityType::PRIVATEUSE1_DRIVER,
};
} // namespace
#endif // USE_KINETO

static_assert(
    std::is_trivial_v<DeviceAndResource>,
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
    activity_t* activity,
    const std::string& key,
    const std::string& value) {
#ifdef USE_KINETO
  activity->addMetadata(key, value);
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
  act.id = static_cast<int32_t>(correlation_id);
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

  bool assertValid() {
    return !config_.profiler_metrics.empty();
  }

  void prepareTraceWithExperimentalOptions(bool add_cpu_activity) {
#ifdef USE_KINETO
    std::set<libkineto::ActivityType> k_activities{
        libkineto::ActivityType::CUDA_PROFILER_RANGE};

    // Only add CPU activities if we are measuring per kernel ranges
    if (add_cpu_activity && config_.profiler_measure_per_kernel) {
      k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
    }

    const size_t num_metrics = config_.profiler_metrics.size();
    std::stringstream configss;

    LOG(INFO) << "CUPTI profiler metrics size = " << num_metrics;

    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
             << "CUPTI_PROFILER_METRICS=";

    for (size_t i = 0; i < num_metrics; i++) {
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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const torch::profiler::impl::ExperimentalConfig& config_;
};
} // namespace

bool collectivesProfilerExists() {
#if defined(KINETO_HAS_HCCL_PROFILER)
  return true;
#endif
  const auto val =
      c10::utils::get_env("TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING");
  return val == "1";
}

#ifdef USE_KINETO
static const std::string setTraceID(const std::string& trace_id) {
  if (trace_id.empty()) {
    return "";
  }
  std::stringstream configss;
  configss << "REQUEST_TRACE_ID=" << trace_id << "\n";
  configss << "REQUEST_GROUP_TRACE_ID=" << trace_id << "\n";
  return configss.str();
}
#endif

void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config,
    const std::string& trace_id) {
#ifdef USE_KINETO
  libkineto::api().resetKinetoTLS();
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  bool has_cpu_activity =
      activities.count(torch::autograd::profiler::ActivityType::CPU);

  if (has_cpu_activity) {
    k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::XPU)) {
    k_activities.insert(kXpuTypes.begin(), kXpuTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::MTIA)) {
    k_activities.insert(kMtiaTypes.begin(), kMtiaTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::HPU)) {
    k_activities.insert(hpuTypes.begin(), hpuTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::CUDA)) {
    k_activities.insert(kCudaTypes.begin(), kCudaTypes.end());
    if (config.enable_cuda_sync_events || get_cuda_sync_enabled()) {
      LOG(INFO) << "Enabling CUDA Sync Events";
      k_activities.insert(libkineto::ActivityType::CUDA_SYNC);
    }
  }
  if (collectivesProfilerExists()) {
    k_activities.insert(libkineto::ActivityType::COLLECTIVE_COMM);
  }
  if (activities.count(torch::autograd::profiler::ActivityType::PrivateUse1)) {
    k_activities.insert(kPrivateUse1Types.begin(), kPrivateUse1Types.end());
  }

  ExperimentalConfigWrapper configWrap(config);

  // Experimental Configuration options are present
  if (config && configWrap.assertValid()) {
    configWrap.prepareTraceWithExperimentalOptions(has_cpu_activity);
    return;
  }

  const std::string configStr = setTraceID(trace_id);

  libkineto::api().activityProfiler().prepareTrace(k_activities, configStr);
#endif // USE_KINETO
}

void toggleCollectionDynamic(const bool enable) {
#ifdef USE_KINETO
  // TODO: We may want to consider adding another input arg for this function
  // if we want to support turning off certain devices and keeping others on.
  // For now, we can keep it simple at have it turn off all tracing of "CUDA"
  // devices
  libkineto::api().activityProfiler().toggleCollectionDynamic(enable);
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

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
#endif // USE_KINETO
}

} // namespace profiler::impl::kineto

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
  // fallthrough
  switch (activity_type) {
    case libkineto::ActivityType::GPU_MEMCPY:
    case libkineto::ActivityType::GPU_MEMSET:
    case libkineto::ActivityType::CONCURRENT_KERNEL:
    case libkineto::ActivityType::CUDA_SYNC:
    case libkineto::ActivityType::GPU_USER_ANNOTATION:
    case libkineto::ActivityType::CUDA_PROFILER_RANGE: {
      // PrivateUse1 kineto backend reuse above ActivityTypes,
      // If PrivateUse1 backend enabled, this should return
      // c10::DeviceType::PrivateUse1.
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::CUDA;
      }();
      return device_type;
    }
    // TODO: T151322015
    case libkineto::ActivityType::MTIA_CCP_EVENTS:
    case libkineto::ActivityType::MTIA_WORKLOADD: {
      // PrivateUse1 kineto backend reuse above ActivityTypes,
      // If PrivateUse1 backend enabled, this should return
      // c10::DeviceType::PrivateUse1.
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::MTIA;
      }();
      return device_type;
    }
    case libkineto::ActivityType::HPU_OP:
      return c10::DeviceType::HPU;
    case libkineto::ActivityType::CPU_OP:
    case libkineto::ActivityType::USER_ANNOTATION:
    case libkineto::ActivityType::EXTERNAL_CORRELATION:
    case libkineto::ActivityType::CUDA_RUNTIME:
    case libkineto::ActivityType::XPU_RUNTIME:
    case libkineto::ActivityType::CPU_INSTANT_EVENT:
    case libkineto::ActivityType::GLOW_RUNTIME:
    case libkineto::ActivityType::MTIA_RUNTIME:
    case libkineto::ActivityType::PYTHON_FUNCTION:
    case libkineto::ActivityType::CUDA_DRIVER:
    case libkineto::ActivityType::PRIVATEUSE1_RUNTIME:
    case libkineto::ActivityType::PRIVATEUSE1_DRIVER:
    case libkineto::ActivityType::OVERHEAD:
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
  libkineto::api().initProfilerIfRegistered();

  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().step();
  } else {
    VLOG(1) << "Profiler is not initialized: skipping step() invocation";
  }
#endif // USE_KINETO
}

} // namespace autograd::profiler

} // namespace torch
