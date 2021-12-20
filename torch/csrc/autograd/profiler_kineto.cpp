#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <limits>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/Context.h>

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
  return torch::profiler::impl::getTime() / 1000;
#endif // USE_KINETO
}
}  // namespace

namespace python_tracer {
namespace {
CallFn call_fn;
TraceEventsFn get_events_fn;
}  // namespace

void registerFunctions(
    CallFn call,
    TraceEventsFn get_events) {
  call_fn = call;
  get_events_fn = get_events;
}

void call(Command c) {
  if (call_fn != nullptr) {
    call_fn(c);
  }
}

std::vector<std::unique_ptr<PyTraceEvent>> get_events() {
  return get_events_fn != nullptr
    ? get_events_fn()
    : std::vector<std::unique_ptr<PyTraceEvent>>();
}

// We do not want `getTimeUs` to be directly visible, but we need a way for
// the python tracer to use the same timing convention as the profiler.
int64_t now() {
  return getTimeUs();
}

struct Replay {
  PyTraceEvent* frame_;
  bool enter_;

  C10_NODISCARD int64_t t() const {
    return enter_ ? frame_->startTime_ : frame_->endTime_;
  }

  C10_NODISCARD size_t idx() const {
    return enter_ ? frame_->call_idx_ : frame_->return_idx_;
  }

  bool operator<(const Replay& other) const {
    return idx() < other.idx();
  }
};

void _push_reverse_order(PyTraceEvent* e, std::vector<std::string>& names) {
  if (e != nullptr) {
    _push_reverse_order(e->parent_, names);
    names.push_back(e->name_);
  }
}
}  // namespace python_tracer

namespace {

// Assumption: Total threads number will not exceed 2^16-1, and total ops will not exceed 2^48 -1.
static inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

struct KinetoThreadLocalState : public torch::profiler::impl::ProfilerThreadLocalStateBase {
  explicit KinetoThreadLocalState(const torch::profiler::impl::ProfilerConfig& config)
    : ProfilerThreadLocalStateBase(config) {
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
      const std::string& evt_name,
      const bool is_async,
      const KinetoObserverContext* ctx) {
    if (!ctx) {
      return;
    }
    auto end_time = ctx->endUS;

    std::lock_guard<std::mutex> guard(state_mutex_);
#ifdef USE_KINETO
    if (cpu_trace) {
      libkineto::GenericTraceActivity op(
          cpu_trace->span,
          libkineto::ActivityType::CPU_OP,
          evt_name);
      op.device = libkineto::processId();
      op.resource = libkineto::systemThreadId();
      op.id = ctx->correlationId;
      op.startTime = ctx->startUs;
      op.endTime = end_time;
      libkineto::api().activityProfiler().recordThreadInfo();
      cpu_trace->activities.emplace_back(std::move(op));
    }
#endif // USE_KINETO

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
        .setAsync(is_async)
        .debugHandle(ctx->debug_handle);
    if (ctx->shapes && !ctx->shapes->empty()) {
      kineto_events_.back().shapes(*ctx->shapes);
    }
    if (ctx->dtypes && !ctx->dtypes->empty()) {
      kineto_events_.back().dtypes(*ctx->dtypes);
    }
    if (ctx->stack && !ctx->stack->empty()) {
      kineto_events_.back().stack(*ctx->stack);
    }
    if (ctx->module_hierarchy) {
      kineto_events_.back().moduleHierarchy(*ctx->module_hierarchy);
    }
    if (ctx->extraArgs && !ctx->extraArgs->empty()) {
      kineto_events_.back().flops(torch::profiler::impl::computeFlops(std::string(evt_name), *ctx->extraArgs));
    }
    kineto_events_.back().cuda_event_start_ = ctx->cuda_event_start_;
    kineto_events_.back().cuda_event_end_ = ctx->cuda_event_end_;
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      std::lock_guard<std::mutex> guard(state_mutex_);
      auto start_time = getTimeUs();
#ifdef USE_KINETO
      if (cpu_trace) {
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
          act.addMetadata(
              "Addr", std::to_string(reinterpret_cast<intptr_t>(ptr)));
          act.addMetadata("Bytes", std::to_string(alloc_size));
          if (total_allocated >= 0) {
            act.addMetadata("Total Allocated", std::to_string(total_allocated));
          }
          if (total_reserved >= 0) {
            act.addMetadata("Total Reserved", std::to_string(total_reserved));
          }
      }
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

  const std::function<void(std::vector<KinetoEvent>&)>& getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(std::function<void(std::vector<KinetoEvent>&)>&& cb) {
    event_post_process_cb_ = std::move(cb);
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
      case libkineto::ActivityType::PYTHON_FUNCTION:
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
          activity.type() != libkineto::ActivityType::USER_ANNOTATION &&
          activity.type() != libkineto::ActivityType::PYTHON_FUNCTION
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

  std::unique_ptr<libkineto::ActivityTraceInterface> finalizeTrace() {
    cpu_trace->span.endTime = getTimeUs();

    // Call events post processing callback before finalizing trace, if there is one.
    if (getEventPostProcessingCallback()) {
      getEventPostProcessingCallback()(kineto_events_);
    }
    finalizeCPUTrace();
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace));
    }

    auto trace = libkineto::api().activityProfiler().stopTrace();
    TORCH_CHECK(trace);
    addTraceEvents(*trace);
    return trace;
  }

  void finalizeCPUTrace() {
    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == kineto_events_.size());
    // startThreadId_seqNum to pointer of activity.
    // Low-16bits of startThreadId and low-48bits seqNum are concatenated into one uint64_t variable as key.
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*> tidSeq2activity;
    uint64_t fwd_bwd_link_id = 1;

    for (size_t idx = 0; idx < cpu_trace->activities.size(); ++idx) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      if (kineto_event.hasShapes()) {
        activity.addMetadata("Input Dims", torch::profiler::impl::shapesToStr(kineto_event.shapes()));
      }
      if (kineto_event.hasStack()) {
        // NB: This is only for the JIT stack. The python stack (if applicable)
        //     is constructed later.
        activity.addMetadata("Call stack", torch::profiler::impl::stacksToStr(kineto_event.stack(), ";"));
      }
      if (kineto_event.hasModuleHierarchy()) {
        activity.addMetadata("Module Hierarchy", torch::profiler::impl::stacksToStr(kineto_event.moduleHierarchy(), "."));
      }
      if (kineto_event.hasTypes()) {
        activity.addMetadata("Input type", torch::profiler::impl::dtypesToStr(kineto_event.dtypes()));
      }
      if (!kineto_event.backend().empty()) {
        activity.addMetadata("Backend", "\"" + kineto_event.backend() + "\"");
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
        generateForwardBackwardLink(kineto_event, fwd_bwd_link_id, activity, tidSeq2activity);
      }
    }

    addPythonEvents();
  }

  void addPythonEvents() {
    auto py_events = python_tracer::get_events();
    for (const auto& e : py_events) {
      TORCH_INTERNAL_ASSERT(
        !e->thread_id_,
        "Profiler expects only single threaded Python tracing.");
    }

    // The remainder of this function merges the Python and Kineto event
    // streams into a single stream. If Python tracing is not enabled, we want
    // to avoid this process altogether to cut down on processing time.
    if (!py_events.size()) {
      return;
    }

    // Kineto event times
    std::vector<int64_t> op_start_times;
    for (const auto& a : cpu_trace->activities) {
      op_start_times.push_back(a.startTime);
    }
    std::sort(op_start_times.begin(), op_start_times.end());

    // Map PyTraceEvent* to sequential integers for JSON export.
    ska::flat_hash_map<python_tracer::PyTraceEvent*, std::string> py_event_indices_ {{ nullptr, std::string("null") }};
    for (size_t i = 0; i < py_events.size(); i++) {
      py_event_indices_.insert({ py_events[i].get(), std::to_string(i) });
    }

    ska::flat_hash_map<std::string, size_t> module_counter_;
    ska::flat_hash_map<size_t, std::string> module_id_map_;
    auto record_module_id = [&](python_tracer::PyTraceEvent* e) {
      if (e->call_type_ == python_tracer::CallType::kPyModuleCall &&
          module_id_map_.find(e->module_id_) == module_id_map_.end()) {
        // We use the fact that operator[] will default initialize new keys.
        module_id_map_[e->module_id_] = std::to_string(module_counter_[e->name_]++);
      }
    };

    // Python events
    std::vector<python_tracer::Replay> py_replay;
    for (const auto& e : py_events) {
      py_replay.push_back({ e.get(), true });
      py_replay.push_back({ e.get(), false });
    }
    std::sort(py_replay.begin(), py_replay.end());

    // In order to determine the state of the python interpreter when a
    // particular op is called, we have to replay the python events and note
    // timestamps which are associated with op start times.
    std::vector<python_tracer::PyTraceEvent*> py_stack;
    ska::flat_hash_map<int64_t, python_tracer::PyTraceEvent*> op_py_map;
    auto replay_it = py_replay.begin();
    for (auto t : op_start_times) {
      while (replay_it != py_replay.end() && replay_it->t() <= t) {
        if (replay_it->enter_) {
          py_stack.push_back(replay_it->frame_);
          record_module_id(replay_it->frame_);
        } else {
          TORCH_INTERNAL_ASSERT(py_stack.size());
          TORCH_INTERNAL_ASSERT(py_stack.back() == replay_it->frame_);
          py_stack.pop_back();
        }
        replay_it++;
      }
      op_py_map.insert({t, py_stack.size() ? py_stack.back() : nullptr});
    }

    auto activities = std::move(cpu_trace->activities);
    auto py_events_it = py_events.begin();
    auto py_device = libkineto::processId();
    auto main_thread = libkineto::systemThreadId();
    auto push_py_event = [&]() {
        auto e = (*py_events_it).get();
        libkineto::GenericTraceActivity op(
          cpu_trace->span,
          libkineto::ActivityType::PYTHON_FUNCTION,
          e->name_
        );

        op.device = py_device;
        op.resource = main_thread;
        op.startTime = e->startTime_;
        op.endTime = e->endTime_;

        op.addMetadata("Python id", py_event_indices_.at(e));
        op.addMetadata("Python parent id", py_event_indices_.at(e->parent_));
        op.addMetadata("Python thread", std::to_string(e->thread_id_));
        if (e->call_type_ == python_tracer::CallType::kPyModuleCall) {
          op.addMetadata("Python module id", module_id_map_.at(e->module_id_));
        }

        cpu_trace->activities.push_back(op);
        py_events_it++;
    };

    TORCH_INTERNAL_ASSERT(activities.size() == kineto_events_.size());
    for (size_t idx = 0; idx < activities.size(); ++idx) {
      auto& activity = activities[idx];

      // Add any python events that occurred between this Kineto event and the
      // previous Kineto event.
      while (py_events_it != py_events.end() && (*py_events_it)->endTime_ <= activity.endTime) {
        push_py_event();
      }

      auto python_caller = op_py_map.at(activity.startTime);
      activity.addMetadata("python_caller_id", py_event_indices_.at(python_caller));

      // If the kineto event has a stack that means the JIT model has a stack
      // associated with it that we need to respect.
      if (!kineto_events_[idx].hasStack()) {
        std::vector<std::string> py_names;
        _push_reverse_order(python_caller, py_names);
        kineto_events_[idx].stack(py_names);
        activity.addMetadata("Call stack", torch::profiler::impl::stacksToStr(py_names, ";"));
      }

      cpu_trace->activities.push_back(activity);
    }

    // Add any Python events which finish after the last Kineto event.
    while (py_events_it != py_events.end()) {
      push_py_event();
    }
  }

  void generateForwardBackwardLink(const KinetoEvent &kineto_event,
    uint64_t &fwd_bwd_link_id,
    libkineto::GenericTraceActivity &activity,
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*> &tidSeq2activity) {
    if (kineto_event.fwdThreadId() > 0) {
      // act is backward op.
      uint64_t key = getForwardThreadKey(kineto_event.fwdThreadId(), kineto_event.sequenceNr());
      auto iter = tidSeq2activity.find(key);
      if (iter != tidSeq2activity.end()) {
        libkineto::GenericTraceActivity* fwd = iter->second;
#ifdef USE_KINETO_UPDATED
        fwd->flow.start = true;
#else
        activity.flow.linkedActivity = fwd; // Only destination side set this, to distinguish with start side.
#endif
        activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
        activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
        ++fwd_bwd_link_id;
      }
    }
    else if (kineto_event.startThreadId() != 0) {
      // act is forward op.
      uint64_t key = getForwardThreadKey(kineto_event.startThreadId(), kineto_event.sequenceNr());
      // Assumption: Among all ops with same sequence number,
      // the one with biggest start time is most likely launching backward op.
      auto iter = tidSeq2activity.find(key);
      if (iter == tidSeq2activity.end()) {
        tidSeq2activity[key] = &activity;
      }
      else {
        // Now the sequence number is only incremented on creating a "Node" object for backward pass,
        // by calling "at::sequence_number::get_and_increment()".
        // Among all ops with same sequence number, the one with biggest startTime is the one launching backward op.
        if (activity.startTime >= iter->second->startTime) {
          tidSeq2activity[key] = &activity;
        }
      }
    }
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_trace;
#endif // USE_KINETO
  uint64_t start_time_;
  std::vector<KinetoEvent> kineto_events_;
  // Optional, if event post-processing is enabled.
  std::function<void(std::vector<KinetoEvent>&)> event_post_process_cb_;
};

KinetoThreadLocalState* getProfilerTLSState() {
  const auto& state = c10::ThreadLocalDebugInfo::get(
      c10::DebugInfoKind::PROFILER_STATE);
  return static_cast<KinetoThreadLocalState*>(state);
}

void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
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
          ctx_ptr->correlationId = corr_id;
          ctx_ptr->startThreadId = at::RecordFunction::currentThreadId();
          ctx_ptr->debug_handle = fn.debugHandle();

          if (config.report_input_shapes) {
            ctx_ptr->shapes = torch::profiler::impl::inputSizes(fn);
            ctx_ptr->dtypes = torch::profiler::impl::inputTypes(fn);
          }

          if (config.with_flops) {
            ctx_ptr->extraArgs = torch::profiler::impl::saveExtraArgs(fn);
          }

          ctx_ptr->sequenceNr = fn.seqNr();
          ctx_ptr->fwdThreadId = fn.forwardThreadId();
          ctx_ptr->recFunScope = (uint8_t)fn.scope();

  #if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
          // backward nodes source range corresponds to the forward node
          // TODO: consider using C++ stack trace
          if (config.with_stack &&
              fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
            auto cs = torch::profiler::impl::prepareCallstack(jit::currentCallstack());
            ctx_ptr->stack = callstackStr(cs);
          }
          if (config.with_modules &&
              fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
            ctx_ptr->module_hierarchy = jit::currentModuleHierarchy();
          }
  #endif
          ctx_ptr->startUs = getTimeUs();
          if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
            try {
              torch::profiler::impl::cudaStubs()->record(nullptr, &ctx_ptr->cuda_event_start_, nullptr);
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to record CUDA event. " << e.what();
            }
          }
          return ctx_ptr;
        } else if (config.state == ProfilerState::NVTX) {
          std::vector<std::vector<int64_t>> shapes;
          if (config.report_input_shapes) {
            shapes = torch::profiler::impl::inputSizes(fn);
          }
          torch::profiler::impl::cudaStubs()->nvtxRangePushA(torch::profiler::impl::getNvtxStr(
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
              torch::profiler::impl::cudaStubs()->record(
                  nullptr, &kineto_ctx_ptr->cuda_event_end_, nullptr);
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to record CUDA event. " << e.what();
            }
          }

          kineto_ctx_ptr->endUS = getTimeUs();
          state_ptr->reportClientActivity(fn.name(), fn.isAsync(), kineto_ctx_ptr);
#ifdef USE_KINETO
          libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
        } else if (config.state == ProfilerState::NVTX) {
          torch::profiler::impl::cudaStubs()->nvtxRangePop();
        }
      })
    .needsInputs(state_ptr->config().report_input_shapes)
    .needsIds(true)
    .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace

void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  auto state_ptr = getProfilerTLSState();
  if (!state_ptr) {
    return;
  }
  auto ctx_ptr = std::make_unique<KinetoObserverContext>();
  ctx_ptr->correlationId = std::numeric_limits<uint64_t>::max();
  ctx_ptr->startThreadId = at::RecordFunction::currentThreadId();
  ctx_ptr->debug_handle = debug_handle;

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */

  ctx_ptr->sequenceNr = -1;
  ctx_ptr->fwdThreadId = ctx_ptr->startThreadId;
  ctx_ptr->recFunScope = (uint8_t)scope;

  ctx_ptr->startUs = start_time_us;
  ctx_ptr->endUS = end_time_us;
  ctx_ptr->endThreadId = at::RecordFunction::currentThreadId();
  state_ptr->reportClientActivity(event_name, false, ctx_ptr.get());
  state_ptr->kineto_events_.back().backend(backend_name);
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
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
    libkineto::ActivityType::PYTHON_FUNCTION,
  };

  std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // also including CUDA_RUNTIME
    libkineto::ActivityType::CUDA_RUNTIME,
  };

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(torch::profiler::impl::ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(torch::profiler::impl::ActivityType::CUDA)) {
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

void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    std::function<void(std::vector<KinetoEvent>&)>&& cb,
    const std::unordered_set<at::RecordScope>& scopes) {
  enableProfiler(config, activities, scopes);
  auto state_ptr = getProfilerTLSState();
  state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  if (config.state != ProfilerState::NVTX) {
    TORCH_CHECK(
        config.state == ProfilerState::KINETO ||
        config.state == ProfilerState::KINETO_GPU_FALLBACK);
    TORCH_CHECK(!activities.empty(), "No activities specified for Kineto profiler");
  } else {
    TORCH_CHECK(torch::profiler::impl::cudaStubs()->enabled(),
        "Can't use NVTX profiler - PyTorch was compiled without CUDA");
  }

  TORCH_CHECK(!profilerEnabled(), "Profiler is already enabled on this thread");
  auto state = std::make_shared<KinetoThreadLocalState>(config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  if (activities.count(torch::profiler::impl::ActivityType::CPU) || config.state == ProfilerState::NVTX) {
    if (config.with_stack) {
      python_tracer::call(python_tracer::Command::kStartOne);
    }
    pushProfilingCallbacks(scopes);
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
    if (config.with_stack) {
      python_tracer::call(python_tracer::Command::kStop);
    }
    at::removeCallback(state_ptr->callbackHandle());
  }

  if (state_ptr->config().state == ProfilerState::NVTX) {
    return std::make_unique<ProfilerResult>();
  }

#ifdef USE_KINETO
  auto trace = state_ptr->finalizeTrace();
#endif
  if (config.with_stack) {
    python_tracer::call(python_tracer::Command::kClear);
  }
#ifdef USE_KINETO
  return std::make_unique<ProfilerResult>(
      state_ptr->start_time_,
      std::move(state_ptr->kineto_events_),
      std::move(trace));
#else
  return std::make_unique<ProfilerResult>(
      std::move(state_ptr->kineto_events_));
#endif // USE_KINETO
}

int64_t KinetoEvent::cudaElapsedUs() const {
  if (!cuda_event_start_ || !cuda_event_end_) {
    return -1;
  }
  try {
    return (int64_t)torch::profiler::impl::cudaStubs()->elapsed(&cuda_event_start_, &cuda_event_end_);
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
