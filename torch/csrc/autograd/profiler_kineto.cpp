#include <torch/csrc/autograd/profiler_kineto.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <sstream>
#include <stdexcept>

#ifdef USE_KINETO
#include <pthread.h>
#include <libkineto.h>

#include <unistd.h>
#include <sys/syscall.h>

// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library when cuPTI is linked statically
// For more details see https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error("Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"

namespace torch { namespace autograd { namespace profiler {

namespace {
// TODO: consider TLS (tid + tls counter)
uint64_t next_correlation_id() {
  static std::atomic<uint64_t> corr_id_ {1};
  return corr_id_++;
}

inline int64_t getTimeUs() {
  using namespace std::chrono;
  return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

// Getting the linux tid is expensive, so cache it.
// Caching linux pids and tids is not advisable in the general case,
// but this is only for profiling purposes and we don't need to handle
// special cases during fork, clone etc.
pid_t cachedTid() {
  static thread_local pid_t tid{(pid_t)syscall(SYS_gettid)};
  return tid;
}

std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes);
std::string stacksToStr(const std::vector<std::string>& stacks);

struct TORCH_API KinetoThreadLocalState : public ProfilerThreadLocalState {
  using ProfilerThreadLocalState::ProfilerThreadLocalState;
  ~KinetoThreadLocalState() override = default;

  void reportClientActivity(
      const at::RecordFunction& fn,
      const KinetoObserverContext* ctx) {
    if (!ctx) {
      return;
    }
    libkineto::ClientTraceActivity op;
    op.startTime = ctx->startUs;
    op.endTime = getTimeUs();
    op.opType = std::string(fn.name().str());
    op.device = 0;
    op.correlation = ctx->correlationId;
    // optimization - postpone shapesToStr till finalizeCPUTrace
    // is called from disableProfiler
    // if (ctx->shapes && !ctx->shapes->empty()) {
    //   op.inputDims = shapesToStr(*ctx->shapes);
    // }

    // Not setting atm
    op.inputTypes = "[]";
    op.arguments = "[]";
    op.outputDims = "[]";
    op.outputTypes = "[]";
    op.inputNames = "[]";
    op.outputNames = "[]";

    // setting both pthread and linux tid for Kineto
    op.sysThreadId = cachedTid();
    op.pthreadId = pthread_self();

    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      kineto_events_.emplace_back();
      kineto_events_.back()
          .activity(op)
          .startThreadId(ctx->startThreadId)
          .endThreadId(ctx->endThreadId)
          .sequenceNr(ctx->sequenceNr)
          .fwdThreadId(ctx->fwdThreadId)
          .scope(ctx->recFunScope);
      if (ctx->shapes && !ctx->shapes->empty()) {
        kineto_events_.back().shapes(*ctx->shapes);
      }
      if (ctx->stack && !ctx->stack->empty()) {
        kineto_events_.back().stack(*ctx->stack);
      }
      if (ctx->extraArgs && !ctx->extraArgs->empty()) {
        kineto_events_.back().flops(computeFlops(std::string(fn.name().str()), *ctx->extraArgs));
      }
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
    }
  }

  void addTraceEvents(libkineto::ActivityTraceInterface& trace) {
    const auto& events = *(trace.activities());
    for (const auto& ev_ptr : events) {
      // ClientTraceActivity events are already processed
      if (ev_ptr->type() != libkineto::ActivityType::CPU_OP) {
        kineto_events_.emplace_back();
        kineto_events_.back()
            .activity(*ev_ptr);
      }
    }
  }

  void finalizeCPUTrace() {
    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == kineto_events_.size());
    for (size_t idx = 0; idx < cpu_trace->activities.size(); ++idx) {
      if (kineto_events_[idx].hasShapes()) {
        cpu_trace->activities[idx].inputDims = shapesToStr(kineto_events_[idx].shapes());
      } else {
        cpu_trace->activities[idx].inputDims = "[]";
      }
      if (kineto_events_[idx].hasStack()) {
        cpu_trace->activities[idx].callStack = stacksToStr(kineto_events_[idx].stack());
      }
    }
  }

  std::vector<KinetoEvent> kineto_events_;
  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_trace =
      std::make_unique<libkineto::CpuTraceBuffer>();
};

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
        if (!state_ptr || state_ptr->config().state != ProfilerState::KINETO) {
          return std::make_unique<KinetoObserverContext>();
        }

        auto corr_id = next_correlation_id();
        libkineto::api().activityProfiler().pushCorrelationId(corr_id);

        auto ctx_ptr = std::make_unique<KinetoObserverContext>();
        ctx_ptr->startUs = getTimeUs();
        ctx_ptr->correlationId = corr_id;
        ctx_ptr->startThreadId = at::RecordFunction::currentThreadId();

        if (state_ptr->config().report_input_shapes) {
          ctx_ptr->shapes = inputSizes(fn);
        }

        if (state_ptr->config().with_flops) {
          ctx_ptr->extraArgs = saveExtraArgs(fn);
        }

        ctx_ptr->sequenceNr = fn.seqNr();
        ctx_ptr->fwdThreadId = fn.forwardThreadId();
        ctx_ptr->recFunScope = (uint8_t)fn.scope();

#ifndef C10_MOBILE
        // backward nodes source range corresponds to the forward node
        // TODO: consider using C++ stack trace
        if (state_ptr->config().with_stack &&
            fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
          auto cs = prepareCallstack(jit::currentCallstack());
          if (cs.empty()) {
            cs = prepareCallstack(jit::tracer::pythonCallstack());
          }
          ctx_ptr->stack = callstackStr(cs);
        }
#endif
        return ctx_ptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state != ProfilerState::KINETO) {
          return;
        }
        auto* kineto_ctx_ptr = static_cast<KinetoObserverContext*>(ctx_ptr);
        TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);

        kineto_ctx_ptr->endThreadId = at::RecordFunction::currentThreadId();

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
  for (size_t t_idx = 0; t_idx < shapes.size(); ++t_idx) {
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

std::string stacksToStr(const std::vector<std::string>& stacks) {
  std::ostringstream oss;
  std::copy(stacks.begin(), stacks.end(), std::ostream_iterator<std::string>(oss, ";"));
  auto rc = oss.str();
  rc.pop_back();
  return rc;
}

} // namespace

void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  TORCH_CHECK(config.state == ProfilerState::KINETO,
      "Supported only in Kineto profiler");

  std::set<libkineto::ActivityType> cpuTypes = {
    libkineto::ActivityType::CPU_OP,
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
  TORCH_CHECK(config.state == ProfilerState::KINETO);
  TORCH_CHECK(!activities.empty(), "No activities specified for Kineto profiler");

  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  auto state = std::make_shared<KinetoThreadLocalState>(config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  state->cpu_trace = std::make_unique<libkineto::CpuTraceBuffer>();
  state->cpu_trace->span.startTime = getTimeUs();
  // TODO: number of GPU ops
  state->cpu_trace->gpuOpCount = -1;
  state->cpu_trace->span.name = "PyTorch Profiler";

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
  TORCH_CHECK(state_ptr && state_ptr->config().state == ProfilerState::KINETO,
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

c10::DeviceType KinetoEvent::deviceType() const {
  // fallthrough
  switch (activity_type_) {
    case (uint8_t)libkineto::ActivityType::GPU_MEMCPY:
    case (uint8_t)libkineto::ActivityType::GPU_MEMSET:
    case (uint8_t)libkineto::ActivityType::CONCURRENT_KERNEL:
      return c10::DeviceType::CUDA;
    case (uint8_t)libkineto::ActivityType::CPU_OP:
    case (uint8_t)libkineto::ActivityType::EXTERNAL_CORRELATION:
    case (uint8_t)libkineto::ActivityType::CUDA_RUNTIME:
      return c10::DeviceType::CPU;
  }
  TORCH_CHECK(false, "Unknown activity type");
}

KinetoEvent::KinetoEvent() : activity_type_((uint8_t)libkineto::ActivityType::CPU_OP) {}

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
