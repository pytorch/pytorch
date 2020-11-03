#include <torch/csrc/autograd/profiler_kineto.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#ifdef USE_KINETO
#include "libkineto.h"

namespace torch { namespace autograd { namespace profiler {

namespace {
// TODO: TLS
std::atomic<uint64_t> corr_id_ {1};
uint64_t next_correlation_id() {
  return corr_id_++;
}

struct TORCH_API KinetoThreadLocalState : public ProfilerThreadLocalState {
  using ProfilerThreadLocalState::ProfilerThreadLocalState;
  virtual ~KinetoThreadLocalState() override = default;

  virtual void reportClientActivity(
      const at::RecordFunction& fn,
      const at::ObserverContext* observer_ctx) override {
    auto ctx = dynamic_cast<const KinetoObserverContext*>(observer_ctx);
    TORCH_CHECK(ctx);
    TORCH_CHECK(config_.state == ProfilerState::KINETO,
        "Supported only in Kineto profiler");
    libkineto::ClientTraceActivity op;
    op.startTime = ctx->startUs;
    op.endTime = (getTime() / 1000);
    op.opType = std::string(fn.name().str());
    op.device = 0; // CPU
    op.correlation = ctx->correlationId;
    if (ctx->shapes && !ctx->shapes->empty()) {
      //op.inputDims = toStr(*ctx->shapes); //
    }
    //op.threadId = pthread_self();

    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      cpu_trace->ops.emplace_back(std::move(op));
      kineto_events_.emplace_back();
      kineto_events_.back()
          .startThreadId(ctx->startThreadId)
          .endThreadId(ctx->endThreadId)
          .sequenceNr(ctx->sequenceNr)
          .fwdThreadId(ctx->fwdThreadId)
          .scope(ctx->recFunScope);
      if (ctx->stack && !ctx->stack->empty()) {
        kineto_events_.back().stack(*ctx->stack);
      }
    }
  }

  std::vector<KinetoEvent> kineto_events_;

  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_trace;
};

KinetoThreadLocalState* getProfilerTLSState() {
  const auto& state = c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE);
  return dynamic_cast<KinetoThreadLocalState*>(state.get());
}

void pushProfilingCallbacks() {
  auto state_ptr = getProfilerTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state != ProfilerState::KINETO) {
          return std::make_unique<KinetoObserverContext>();
        }

        auto corr_id = next_correlation_id();
        libkineto::api().pushCorrelationId(corr_id);

        auto ctx_ptr = std::make_unique<KinetoObserverContext>();
        ctx_ptr->startUs = getTime() / 1000;
        ctx_ptr->correlationId = corr_id;
        ctx_ptr->startThreadId = at::RecordFunction::currentThreadId();

        if (state_ptr->config().report_input_shapes) {
          ctx_ptr->shapes = inputSizes(fn);
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
        auto kineto_ctx_ptr = dynamic_cast<KinetoObserverContext*>(ctx_ptr);
        TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);

        kineto_ctx_ptr->endThreadId = at::RecordFunction::currentThreadId();

        state_ptr->reportClientActivity(fn, kineto_ctx_ptr);
        libkineto::api().popCorrelationId();
      })
    .needsInputs(state_ptr->config().report_input_shapes)
    .needsIds(true));
  state_ptr->setCallbackHandle(handle);
}

} // namespace

void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  TORCH_CHECK(config.state == ProfilerState::KINETO,
      "Supported only in Kineto profiler");

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(ActivityType::CPU)) {
    k_activities.insert(libkineto::ActivityType::EXTERNAL_CORRELATION);
    k_activities.insert(libkineto::ActivityType::CUDA_RUNTIME);
  }
  //if (activities.count(ActivityType::CUDA_RUNTIME)) {
  //  k_activities.insert(libkineto::ActivityType::CUDA_RUNTIME);
  //}
  if (activities.count(ActivityType::CUDA)) {
    k_activities.insert(libkineto::ActivityType::GPU_MEMCPY);
    k_activities.insert(libkineto::ActivityType::GPU_MEMSET);
    k_activities.insert(libkineto::ActivityType::CONCURRENT_KERNEL);
  }

  //if (!libkineto::api().hasProfilerRegistered()) {
  //  libkineto::api().registerProfiler(
  //    std::make_unique<libkineto::ActivityProfilerInterface>(false));
  //}
  //libkineto::api().initProfilerIfRegistered();
  libkineto::api().prepareTrace(k_activities);
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
  state->cpu_trace->span.startTime = getTime() / 1000;
  state->cpu_trace->gpuOpCount = -1;
  state->cpu_trace->span.name = "PyTorch Profiler";

  if (activities.count(ActivityType::CPU)) {
    pushProfilingCallbacks();
  }

  if (!libkineto::api().traceActive()) {
    libkineto::api().startTrace();
  }

  state->mark("__start_profile", false);
}

std::vector<std::vector<KinetoEvent>> filterTrace(
    std::unique_ptr<libkineto::ActivityTraceInterface>&& trace) {
  // tbd
  return std::vector<std::vector<KinetoEvent>>();
}

ProfilerResult disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use DebugInfoGuard
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<KinetoThreadLocalState*>(state.get());
  TORCH_CHECK(state_ptr && state_ptr->config().state == ProfilerState::KINETO,
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->callbackHandle() > 0) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  state_ptr->mark("__stop_profile");

  state_ptr->cpu_trace->span.endTime = getTime() / 1000;

  libkineto::api().transferCpuTrace(std::move(state_ptr->cpu_trace));

  std::vector<std::vector<KinetoEvent>> kineto_events = filterTrace(
      std::move(libkineto::api().stopTrace()));
  auto legacy_events = state_ptr->consolidate();
  return ProfilerResult(kineto_events, legacy_events);
}

KinetoEvent& KinetoEvent::activity(const libkineto::TraceActivity& activity) {
  name_ = activity.name();
  device_index_ = activity.deviceId();
  start_us_ = activity.timestamp();
  duration_us_ = activity.duration();
  correlation_id_ = activity.correlationId();
  return *this;
}
#endif

bool kinetoAvailable() {
#ifdef USE_KINETO
  return true;
#else
  return false;
#endif
}

}}}
