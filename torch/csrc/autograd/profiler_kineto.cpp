#include <c10/util/irange.h>
#include <torch/csrc/autograd/profiler_kineto.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <sstream>
#include <stdexcept>

#ifdef USE_KINETO
#include <libkineto.h>
#include <time_since_epoch.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library when cuPTI is linked statically
// For more details see https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error("Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch { namespace autograd { namespace profiler {

namespace {
const std::string kMemoryEventName = "[memory]";
// TODO: consider TLS (tid + tls counter)
uint64_t next_correlation_id() {
  static std::atomic<uint64_t> corr_id_ {1};
  return corr_id_++;
}

inline int64_t getTimeUs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return getTime() / 1000;
#endif // USE_KINETO
}

std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes);
std::string stacksToStr(const std::vector<std::string>& stacks);
std::string dtypesToStr(const std::vector<std::string>& types);
std::vector<std::string> inputTypes(const at::RecordFunction& fn);

struct KinetoThreadLocalState : public ProfilerThreadLocalState {
  explicit KinetoThreadLocalState(const ProfilerConfig& config)
    : ProfilerThreadLocalState(config) {
    start_time_ = getTimeUs();
#ifdef USE_KINETO
    cpu_trace = std::make_unique<libkineto::CpuTraceBuffer>();
    cpu_trace->span.startTime = start_time_;
    cpu_trace->gpuOpCount = -1;
    cpu_trace->span.name = "PyTorch Profiler";
#endif // USE_KINETO
  }
  ~KinetoThreadLocalState() override = default;

  void reportClientActivity(
      const at::RecordFunction& fn,
      const KinetoObserverContext* ctx) {
    if (!ctx) {
      return;
    }
    std::string evt_name(fn.name().str());
    auto end_time = getTimeUs();
#ifdef USE_KINETO
    libkineto::GenericTraceActivity op(
        cpu_trace->span,
        libkineto::ActivityType::CPU_OP,
        evt_name);
    op.device = libkineto::processId();
    op.resource = libkineto::systemThreadId();
    op.id = ctx->correlationId;
    op.startTime = ctx->startUs;
    op.endTime = end_time;
    // optimization - postpone shapesToStr till finalizeCPUTrace
    // is called from disableProfiler
    // if (ctx->shapes && !ctx->shapes->empty()) {
    //   op.inputDims = shapesToStr(*ctx->shapes);
    // }

    libkineto::api().activityProfiler().recordThreadInfo();
#endif // USE_KINETO
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      kineto_events_.emplace_back();
      kineto_events_.back()
          .name(evt_name)
          .startUs(ctx->startUs)
          .durationUs(end_time - ctx->startUs)
          .correlationId(ctx->correlationId)
          .deviceType(c10::DeviceType::CPU)
          .startThreadId(ctx->startThreadId)
          .endThreadId(ctx->endThreadId)
          .sequenceNr(ctx->sequenceNr)
          .fwdThreadId(ctx->fwdThreadId)
          .scope(ctx->recFunScope)
          .setAsync(fn.isAsync());
      if (ctx->shapes && !ctx->shapes->empty()) {
        kineto_events_.back().shapes(*ctx->shapes);
      }
      if (ctx->dtypes && !ctx->dtypes->empty()) {
        kineto_events_.back().dtypes(*ctx->dtypes);
      }
      if (ctx->stack && !ctx->stack->empty()) {
        kineto_events_.back().stack(*ctx->stack);
      }
      if (ctx->extraArgs && !ctx->extraArgs->empty()) {
        kineto_events_.back().flops(computeFlops(std::string(fn.name().str()), *ctx->extraArgs));
      }
      kineto_events_.back().cuda_event_start_ = ctx->cuda_event_start_;
      kineto_events_.back().cuda_event_end_ = ctx->cuda_event_end_;
#ifdef USE_KINETO
      cpu_trace->activities.emplace_back(std::move(op));
#endif // USE_KINETO
    }
  }

  // TODO: use kineto
  void reportMemoryUsage(
      void* /* unused */,
      int64_t alloc_size,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      std::lock_guard<std::mutex> guard(state_mutex_);
      auto start_time = getTimeUs();
#ifdef USE_KINETO
      libkineto::api().activityProfiler().recordThreadInfo();

      cpu_trace->activities.emplace_back(
          libkineto::GenericTraceActivity(
            cpu_trace->span,
            libkineto::ActivityType::CPU_INSTANT_EVENT,
            kMemoryEventName));
      auto& act = cpu_trace->activities.back();
      act.device = libkineto::processId();
      act.resource = libkineto::systemThreadId();

      act.startTime = start_time;
      act.addMetadata("Device Type", std::to_string((int8_t)device.type()));
      act.addMetadata("Device Id", std::to_string(device.index()));
      act.addMetadata("Bytes", std::to_string(alloc_size));
#endif // USE_KINETO

      kineto_events_.emplace_back();
      auto& evt = kineto_events_.back();
      evt.name(kMemoryEventName)
        .startUs(start_time)
        .deviceIndex(device.index())
        .deviceType(device.type())
        .nBytes(alloc_size)
        .startThreadId(at::RecordFunction::currentThreadId());
    }
  }

#ifdef USE_KINETO
  c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
    // fallthrough
    switch (activity_type) {
      case libkineto::ActivityType::GPU_MEMCPY:
      case libkineto::ActivityType::GPU_MEMSET:
      case libkineto::ActivityType::CONCURRENT_KERNEL:
      case libkineto::ActivityType::GPU_USER_ANNOTATION:
        return c10::DeviceType::CUDA;
      case libkineto::ActivityType::CPU_OP:
      case libkineto::ActivityType::USER_ANNOTATION:
      case libkineto::ActivityType::EXTERNAL_CORRELATION:
      case libkineto::ActivityType::CUDA_RUNTIME:
      case libkineto::ActivityType::CPU_INSTANT_EVENT:
      case libkineto::ActivityType::GLOW_RUNTIME:
        return c10::DeviceType::CPU;
      default: {
        LOG(WARNING) << "Unknown activity type (" << (uint8_t)activity_type
                    << "), assuming CPU device";
        return c10::DeviceType::CPU;
      }
    }
  }

  void addTraceEvents(libkineto::ActivityTraceInterface& trace) {
    const auto& events = *(trace.activities());
    for (const auto& ev_ptr : events) {
      const auto& activity = *ev_ptr;
      // These events are already processed
      if (activity.type() != libkineto::ActivityType::CPU_OP &&
          activity.type() != libkineto::ActivityType::CPU_INSTANT_EVENT &&
          activity.type() != libkineto::ActivityType::USER_ANNOTATION
      ) {
        kineto_events_.emplace_back();
        auto& kineto_event = kineto_events_.back();
        kineto_event.name(activity.name())
          .deviceIndex(activity.deviceId())
          .deviceResourceId(activity.resourceId())
          .startUs(activity.timestamp())
          .durationUs(activity.duration())
          .activityType((uint8_t)activity.type());
        if (activity.linkedActivity()) {
          kineto_event.linkedCorrelationId(
              activity.linkedActivity()->correlationId());
        }
        kineto_event.deviceType(deviceTypeFromActivity(activity.type()));
      }
    }
  }

  void finalizeCPUTrace() {
    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == kineto_events_.size());
    for (size_t idx = 0; idx < cpu_trace->activities.size(); ++idx) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      if (kineto_event.hasShapes()) {
        activity.addMetadata("Input Dims", shapesToStr(kineto_event.shapes()));
      }
      if (kineto_event.hasStack()) {
        activity.addMetadata("Call stack", stacksToStr(kineto_event.stack()));
      }
      if (kineto_event.hasTypes()) {
        activity.addMetadata("Input type", dtypesToStr(kineto_event.dtypes()));
      }

      // add information about an associated forward op, if a sequence number
      // is available (e.g. during training)
      if (kineto_event.sequenceNr() >= 0) {
        activity.addMetadata(
            "Fwd thread id",
            std::to_string(kineto_event.fwdThreadId()));
        activity.addMetadata(
            "Sequence number",
            std::to_string(kineto_event.sequenceNr()));
      }
    }
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_trace;
#endif // USE_KINETO
  uint64_t start_time_;
  std::vector<KinetoEvent> kineto_events_;
};

std::vector<std::string> inputTypes(const at::RecordFunction& fn) {
  std::vector<std::string> types;
  types.reserve(fn.inputs().size());
  for (const c10::IValue& input : fn.inputs()) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        types.push_back(
            static_cast<std::string>(input.toTensor().dtype().name()));
      } else {
        types.emplace_back();
      }
    } else if (input.isScalar() || input.isList()) {
      types.push_back(input.tagKind());
    } else {
      types.emplace_back();
    }
  }
  return types;
}

KinetoThreadLocalState* getProfilerTLSState() {
  const auto& state = c10::ThreadLocalDebugInfo::get(
      c10::DebugInfoKind::PROFILER_STATE);
  return static_cast<KinetoThreadLocalState*>(state);
}

void pushProfilingCallbacks() {
  auto state_ptr = getProfilerTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr) {
          return nullptr;
        }
        const auto& config = state_ptr->config();
        if (config.state == ProfilerState::KINETO ||
            config.state == ProfilerState::KINETO_GPU_FALLBACK) {
          auto corr_id = next_correlation_id();
#ifdef USE_KINETO
          libkineto::api().activityProfiler().pushCorrelationId(corr_id);
#endif // USE_KINETO

          auto ctx_ptr = std::make_unique<KinetoObserverContext>();
          ctx_ptr->startUs = getTimeUs();
          ctx_ptr->correlationId = corr_id;
          ctx_ptr->startThreadId = at::RecordFunction::currentThreadId();

          if (config.report_input_shapes) {
            ctx_ptr->shapes = inputSizes(fn);
            ctx_ptr->dtypes = inputTypes(fn);
          }

          if (config.with_flops) {
            ctx_ptr->extraArgs = saveExtraArgs(fn);
          }

          ctx_ptr->sequenceNr = fn.seqNr();
          ctx_ptr->fwdThreadId = fn.forwardThreadId();
          ctx_ptr->recFunScope = (uint8_t)fn.scope();

  #if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
          // backward nodes source range corresponds to the forward node
          // TODO: consider using C++ stack trace
          if (config.with_stack &&
              fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
            auto cs = prepareCallstack(jit::currentCallstack());
            if (cs.empty()) {
              cs = prepareCallstack(jit::tracer::pythonCallstack());
            }
            ctx_ptr->stack = callstackStr(cs);
          }
  #endif
          if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
            try {
              cudaStubs()->record(nullptr, &ctx_ptr->cuda_event_start_, nullptr);
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to record CUDA event. " << e.what();
            }
          }
          return ctx_ptr;
        } else if (config.state == ProfilerState::NVTX) {
          std::vector<std::vector<int64_t>> shapes;
          if (config.report_input_shapes) {
            shapes = inputSizes(fn);
          }
          cudaStubs()->nvtxRangePushA(getNvtxStr(
            fn.name(), fn.seqNr(), shapes).c_str());
        }
        return nullptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr) {
          return;
        }
        const auto& config = state_ptr->config();
        if (config.state == ProfilerState::KINETO ||
            config.state == ProfilerState::KINETO_GPU_FALLBACK) {
          auto* kineto_ctx_ptr = static_cast<KinetoObserverContext*>(ctx_ptr);
          TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);

          kineto_ctx_ptr->endThreadId = at::RecordFunction::currentThreadId();

          if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
            try {
              cudaStubs()->record(
                  nullptr, &kineto_ctx_ptr->cuda_event_end_, nullptr);
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to record CUDA event. " << e.what();
            }
          }

          state_ptr->reportClientActivity(fn, kineto_ctx_ptr);
#ifdef USE_KINETO
          libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
        } else if (config.state == ProfilerState::NVTX) {
          cudaStubs()->nvtxRangePop();
        }
      })
    .needsInputs(state_ptr->config().report_input_shapes)
    .needsIds(true));
  state_ptr->setCallbackHandle(handle);
}

std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes) {
  std::ostringstream oss;
  oss << "[";
  for (const auto t_idx : c10::irange(shapes.size())) {
    if (t_idx > 0) {
      oss << ", ";
    }
    oss << "[";
    for (size_t s_idx = 0; s_idx < shapes[t_idx].size(); ++s_idx) {
      if (s_idx > 0) {
        oss << ", ";
      }
      oss << shapes[t_idx][s_idx];
    }
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

std::string dtypesToStr(const std::vector<std::string>& types) {
  if (types.empty()) {
    return "[]";
  } else {
    std::ostringstream oss;
    std::transform(
        types.begin(),
        types.end(),
        std::ostream_iterator<std::string>(oss, ", "),
        [](std::string s) -> std::string { return "\"" + s + "\""; });
    auto rc = oss.str();
    rc.erase(rc.length() - 2); // remove last ", "
    return "[" + rc + "]";
  }
}

std::string stacksToStr(const std::vector<std::string>& stacks) {
  std::ostringstream oss;
  std::transform(
      stacks.begin(),
      stacks.end(),
      std::ostream_iterator<std::string>(oss, ";"),
      [](std::string s) -> std::string {
#ifdef _WIN32
        // replace the windows backslash with forward slash
        std::replace(s.begin(), s.end(), '\\', '/');
#endif
        return s;
      });
  auto rc = oss.str();
  rc.pop_back();
  return "\"" + rc + "\"";
}

} // namespace

void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");
#ifdef USE_KINETO
  std::set<libkineto::ActivityType> cpuTypes = {
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,
  };

  std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // also including CUDA_RUNTIME
    libkineto::ActivityType::CUDA_RUNTIME,
  };

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(ActivityType::CUDA)) {
    k_activities.insert(cudaTypes.begin(), cudaTypes.end());
  }

  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/!at::hasCUDA(), /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  libkineto::api().activityProfiler().prepareTrace(k_activities);
#endif // USE_KINETO
}

void enableProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  if (config.state != ProfilerState::NVTX) {
    TORCH_CHECK(
        config.state == ProfilerState::KINETO ||
        config.state == ProfilerState::KINETO_GPU_FALLBACK);
    TORCH_CHECK(!activities.empty(), "No activities specified for Kineto profiler");
  } else {
    TORCH_CHECK(cudaStubs()->enabled(),
        "Can't use NVTX profiler - PyTorch was compiled without CUDA");
  }

  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  auto state = std::make_shared<KinetoThreadLocalState>(config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  if (activities.count(ActivityType::CPU) || config.state == ProfilerState::NVTX) {
    pushProfilingCallbacks();
  }

#ifdef USE_KINETO
  if (config.state != ProfilerState::NVTX) {
    libkineto::api().activityProfiler().startTrace();
  }
#endif // USE_KINETO
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use DebugInfoGuard
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<KinetoThreadLocalState*>(state.get());
  const auto& config = state_ptr->config();
  TORCH_CHECK(state_ptr && (
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::NVTX),
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  if (state_ptr->config().state == ProfilerState::NVTX) {
    return std::make_unique<ProfilerResult>();
  }

#ifdef USE_KINETO
  state_ptr->cpu_trace->span.endTime = getTimeUs();
  state_ptr->finalizeCPUTrace();
  libkineto::api().activityProfiler().transferCpuTrace(std::move(state_ptr->cpu_trace));

  auto trace = libkineto::api().activityProfiler().stopTrace();
  TORCH_CHECK(trace);
  state_ptr->addTraceEvents(*trace);
  return std::make_unique<ProfilerResult>(
      state_ptr->start_time_,
      std::move(state_ptr->kineto_events_),
      std::move(trace));
#else
  return std::make_unique<ProfilerResult>(
      std::move(state_ptr->kineto_events_));
#endif // USE_KINETO
}

void addMetadataJson(const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
#endif // USE_KINETO
}

int64_t KinetoEvent::cudaElapsedUs() const {
  if (!cuda_event_start_ || !cuda_event_end_) {
    return -1;
  }
  try {
    return (int64_t)cudaStubs()->elapsed(&cuda_event_start_, &cuda_event_end_);
  } catch (std::exception& e) {
    LOG(WARNING) << "Failed to measure time between two CUDA events. "
        << e.what();
  }
  return -1;
}

#ifdef USE_KINETO
ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    std::unique_ptr<libkineto::ActivityTraceInterface> trace)
  : trace_start_us_(start_time),
    events_(std::move(events)),
    trace_(std::move(trace)) {}
#else
ProfilerResult::ProfilerResult(std::vector<KinetoEvent> events)
  : events_(std::move(events)) {}
#endif // USE_KINETO
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

#ifdef USE_KINETO
void ProfilerResult::save(const std::string& path) {
  // Kineto's save is destructive
  TORCH_CHECK(!saved_, "Trace is already saved");
  trace_->save(path);
  saved_ = true;
}
#endif // USE_KINETO

}}} // namespace torch::autograd::profiler
