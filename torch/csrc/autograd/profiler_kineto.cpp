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
#endif

namespace torch { namespace autograd { namespace profiler {

namespace {
// TODO: consider TLS (tid + tls counter)
uint64_t next_correlation_id() {
  static std::atomic<uint64_t> corr_id_ {1};
  return corr_id_++;
}

inline int64_t getTimeUs() {
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
}

std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes);
std::string stacksToStr(const std::vector<std::string>& stacks);
std::string dtypesToStr(const std::vector<std::string>& types);
std::vector<std::string> inputTypes(const at::RecordFunction& fn);

struct KinetoThreadLocalState : public ProfilerThreadLocalState {
  explicit KinetoThreadLocalState(const ProfilerConfig& config)
    : ProfilerThreadLocalState(config) {
    cpu_trace = std::make_unique<libkineto::CpuTraceBuffer>();
    cpu_trace->span.startTime = getTimeUs();
    cpu_trace->gpuOpCount = -1;
    cpu_trace->span.name = "PyTorch Profiler";
  }
  ~KinetoThreadLocalState() override = default;

  void reportClientActivity(
      const at::RecordFunction& fn,
      const KinetoObserverContext* ctx) {
    if (!ctx) {
      return;
    }

    libkineto::GenericTraceActivity op(
        cpu_trace->span,
        libkineto::ActivityType::CPU_OP,
        std::string(fn.name().str()));
    op.device = libkineto::processId();
    op.resource = libkineto::systemThreadId();
    op.id = ctx->correlationId;
    op.startTime = ctx->startUs;
    op.endTime = getTimeUs();
    // optimization - postpone shapesToStr till finalizeCPUTrace
    // is called from disableProfiler
    // if (ctx->shapes && !ctx->shapes->empty()) {
    //   op.inputDims = shapesToStr(*ctx->shapes);
    // }

    libkineto::api().activityProfiler().recordThreadInfo();

    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      kineto_events_.emplace_back();
      kineto_events_.back()
          .activity(op)
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
      cpu_trace->activities.emplace_back(std::move(op));
    }
  }

  // TODO: use kineto
  void reportMemoryUsage(
      void* /* unused */,
      int64_t alloc_size,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      uint64_t thread_id = at::RecordFunction::currentThreadId();
      LegacyEvent evt(
          EventKind::MemoryAlloc,
          at::StringView(""),
          thread_id,
          config_.state == ProfilerState::CUDA);
      evt.setCpuUs(getTimeUs()); // upd. time using Kineto's clock
      evt.updateMemoryStats(alloc_size, device);
      getEventList(thread_id).record(std::move(evt));

      {
        std::lock_guard<std::mutex> guard(state_mutex_);
        libkineto::api().activityProfiler().recordThreadInfo();

        memory_events_.emplace_back(
            cpu_trace->span,
            libkineto::ActivityType::CPU_INSTANT_EVENT,
            "[memory]");
        auto& act = memory_events_.back();
        act.device = libkineto::processId();
        act.resource = libkineto::systemThreadId();

        act.startTime = getTimeUs();
        act.addMetadata("Device Type", std::to_string((int8_t)device.type()));
        act.addMetadata("Device Id", std::to_string(device.index()));
        act.addMetadata("Bytes", std::to_string(alloc_size));
      }
    }
  }

  void addTraceEvents(libkineto::ActivityTraceInterface& trace) {
    const auto& events = *(trace.activities());
    for (const auto& ev_ptr : events) {
      // These events are already processed
      if (ev_ptr->type() != libkineto::ActivityType::CPU_OP &&
          ev_ptr->type() != libkineto::ActivityType::CPU_INSTANT_EVENT
          && ev_ptr->type() != libkineto::ActivityType::USER_ANNOTATION
      ) {
        kineto_events_.emplace_back();
        kineto_events_.back()
            .activity(*ev_ptr);
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

    cpu_trace->activities.insert(
      cpu_trace->activities.end(),
      memory_events_.begin(),
      memory_events_.end());
  }

  std::vector<KinetoEvent> kineto_events_;
  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_trace;
  std::vector<libkineto::GenericTraceActivity> memory_events_;
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
        if (config.state != ProfilerState::KINETO &&
            config.state != ProfilerState::KINETO_GPU_FALLBACK) {
          return nullptr;
        }

        auto corr_id = next_correlation_id();
        libkineto::api().activityProfiler().pushCorrelationId(corr_id);

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
            C10_LOG_EVERY_N(WARNING, 1000) << "Failed to record CUDA event. "
                                           << e.what();
          }
        }
        return ctx_ptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr) {
          return;
        }
        const auto& config = state_ptr->config();
        if (config.state != ProfilerState::KINETO &&
            config.state != ProfilerState::KINETO_GPU_FALLBACK) {
          return;
        }
        auto* kineto_ctx_ptr = static_cast<KinetoObserverContext*>(ctx_ptr);
        TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);

        kineto_ctx_ptr->endThreadId = at::RecordFunction::currentThreadId();

        if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
          try {
            cudaStubs()->record(
                nullptr, &kineto_ctx_ptr->cuda_event_end_, nullptr);
          } catch (const std::exception& e) {
            C10_LOG_EVERY_N(WARNING, 1000) << "Failed to record CUDA event. "
                                           << e.what();
          }
        }

        state_ptr->reportClientActivity(fn, kineto_ctx_ptr);
        libkineto::api().activityProfiler().popCorrelationId();
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
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");

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
}

void enableProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK);
  TORCH_CHECK(!activities.empty(), "No activities specified for Kineto profiler");

  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  auto state = std::make_shared<KinetoThreadLocalState>(config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  if (activities.count(ActivityType::CPU)) {
    pushProfilingCallbacks();
  }

  libkineto::api().activityProfiler().startTrace();

  state->mark("__start_profile", false);
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use DebugInfoGuard
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<KinetoThreadLocalState*>(state.get());
  const auto& config = state_ptr->config();
  TORCH_CHECK(state_ptr && (
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK),
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  state_ptr->mark("__stop_profile", false);

  state_ptr->cpu_trace->span.endTime = getTimeUs();

  state_ptr->finalizeCPUTrace();
  libkineto::api().activityProfiler().transferCpuTrace(std::move(state_ptr->cpu_trace));

  auto trace = libkineto::api().activityProfiler().stopTrace();
  TORCH_CHECK(trace);
  state_ptr->addTraceEvents(*trace);
  return std::make_unique<ProfilerResult>(
      std::move(state_ptr->kineto_events_),
      state_ptr->consolidate(),
      std::move(trace));
}

void addMetadataJson(const std::string& key, const std::string& value) {
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
}

KinetoEvent& KinetoEvent::activity(const libkineto::TraceActivity& activity) {
  name_ = activity.name();
  device_index_ = activity.deviceId();
  device_resource_id_ = activity.resourceId();
  start_us_ = activity.timestamp();
  duration_us_ = activity.duration();
  // Set the correlation id for the PyTorch CPU ops.
  // Note: skip setting the correlation ids for other activities to avoid
  // an incorrect attribution of CUDA kernels.
  if (activity.type() == libkineto::ActivityType::CPU_OP) {
    correlation_id_ = activity.correlationId();
  }
  activity_type_ = (uint8_t)activity.type();
  if (activity.linkedActivity()) {
    linked_correlation_id_ = activity.linkedActivity()->correlationId();
  }
  return *this;
}

int64_t KinetoEvent::cudaElapsedUs() const {
  if (!cuda_event_start_ || !cuda_event_end_) {
    return -1;
  }
  try {
    return (int64_t)cudaStubs()->elapsed(&cuda_event_start_, &cuda_event_end_);
  } catch (std::exception& e) {
    C10_LOG_EVERY_N(WARNING, 1000)
        << "Failed to measure time between two CUDA events. "
        << e.what();
  }
  return -1;
}

c10::DeviceType KinetoEvent::deviceType() const {
  // fallthrough
  switch (activity_type_) {
    case (uint8_t)libkineto::ActivityType::GPU_MEMCPY:
    case (uint8_t)libkineto::ActivityType::GPU_MEMSET:
    case (uint8_t)libkineto::ActivityType::CONCURRENT_KERNEL:
    case (uint8_t)libkineto::ActivityType::GPU_USER_ANNOTATION:
      return c10::DeviceType::CUDA;
    case (uint8_t)libkineto::ActivityType::CPU_OP:
    case (uint8_t)libkineto::ActivityType::USER_ANNOTATION:
    case (uint8_t)libkineto::ActivityType::EXTERNAL_CORRELATION:
    case (uint8_t)libkineto::ActivityType::CUDA_RUNTIME:
    case (uint8_t)libkineto::ActivityType::CPU_INSTANT_EVENT:
      return c10::DeviceType::CPU;
  }
  TORCH_CHECK(false, "Unknown activity type");
}

ProfilerResult::ProfilerResult(
    std::vector<KinetoEvent> events,
    thread_event_lists legacy_events,
    std::unique_ptr<libkineto::ActivityTraceInterface> trace)
  : events_(std::move(events)),
    legacy_events_(std::move(legacy_events)),
    trace_(std::move(trace)) {}
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  // Kineto's save is destructive
  TORCH_CHECK(!saved_, "Trace is already saved");
  trace_->save(path);
  saved_ = true;
}

}}} // namespace torch::autograd::profiler
#endif /* USE_KINETO */
