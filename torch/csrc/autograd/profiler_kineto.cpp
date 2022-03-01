#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/nvtx_observer.h>

#include <ATen/Context.h>

#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef USE_KINETO
#include <libkineto.h>
#include <time_since_epoch.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch {
namespace autograd {
namespace profiler {

namespace {
const std::string kMemoryEventName = "[memory]";
// TODO: consider TLS (tid + tls counter)
uint64_t next_correlation_id() {
  static std::atomic<uint64_t> corr_id_{1};
  return corr_id_++;
}

inline int64_t getTimeUs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return torch::profiler::impl::getTime() / 1000;
#endif // USE_KINETO
}
} // namespace

namespace python_tracer {
namespace {
CallFn call_fn;
TraceEventsFn get_events_fn;
} // namespace

void registerFunctions(CallFn call, TraceEventsFn get_events) {
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
} // namespace python_tracer

namespace {
using torch::profiler::impl::ProfilerThreadLocalStateBase;
using torch::profiler::impl::ActiveProfilerType;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct OpEventData {
    // POD members
    int64_t start_us_;
    int64_t end_us_;
    uint64_t correlation_id_;
    uint64_t start_thread_id_;
    uint64_t end_thread_id_;
    int64_t sequence_number_;
    uint64_t forward_thread_id_;
    uint8_t record_function_scope_;
    bool is_async_;
    int64_t debug_handle_;
    torch::profiler::impl::kineto::DeviceAndResource kineto_info_;

    std::string name_;

    // report_input_shapes
    std::vector<std::vector<int64_t>> shapes_;
    std::vector<std::string> dtypes_;

    // with_stack
    std::vector<std::string> stack_;

    // with_modules
    c10::optional<std::vector<std::string>> module_hierarchy_;

    // with_flops
    std::unordered_map<std::string, c10::IValue> extra_args_;

    // reportBackendEventToActiveKinetoProfiler
    c10::optional<std::string> backend_;

    // ProfilerState::KINETO_GPU_FALLBACK
    torch::profiler::impl::CUDAEventStub cuda_event_start_ = nullptr;
    torch::profiler::impl::CUDAEventStub cuda_event_end_ = nullptr;
};

struct MemoryEventData {
  int64_t start_time;
  void* ptr;
  int64_t alloc_size;
  int64_t total_allocated;
  int64_t total_reserved;
  uint64_t threadID;
  torch::profiler::impl::kineto::DeviceAndResource kineto_info;
  c10::DeviceType device_type;
  c10::DeviceIndex device_index;
};
static_assert(std::is_pod<MemoryEventData>::value, "Non-POD member of MemoryEventData.");

// Assumption: Total threads number will not exceed 2^16-1, and total ops will
// not exceed 2^48 -1.
static inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

struct KinetoObserverContext : public at::ObserverContext {
  explicit KinetoObserverContext(OpEventData* data) : data_(data) {}
  OpEventData* data_;
};

struct KinetoThreadLocalState : public ProfilerThreadLocalStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerThreadLocalStateBase(config),
        start_time_(getTimeUs()),
        activities_(std::move(activities)),
        cpu_trace_(start_time_, "PyTorch Profiler") {}
  ~KinetoThreadLocalState() override = default;

  static KinetoThreadLocalState* getTLS() {
    auto tls = ProfilerThreadLocalStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(tls);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  bool tracePython() {
    return config().with_stack && activities_.count(ActivityType::CPU);
  }

  std::unique_ptr<KinetoObserverContext> newOpEvent() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    op_events_.emplace_back();
    return std::make_unique<KinetoObserverContext>(&op_events_.back());
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      memory_events_.push_back(
          {getTimeUs(),
           ptr,
           alloc_size,
           total_allocated,
           total_reserved,
           at::RecordFunction::currentThreadId(),
           torch::profiler::impl::kineto::kineto_ids(),
           device.type(),
           device.index()});
    }
  }

  const std::function<void(std::vector<KinetoEvent>&)>&
  getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(
      std::function<void(std::vector<KinetoEvent>&)>&& cb) {
    event_post_process_cb_ = std::move(cb);
  }

  torch::profiler::impl::kineto::ActivityTraceWrapper finalizeTrace() {
    auto end_time = getTimeUs();
    materializeOpEvents();

    // Call events post processing callback before finalizing trace, if there is
    // one.
    if (getEventPostProcessingCallback()) {
      getEventPostProcessingCallback()(kineto_events_);
    }

    finalizeCPUTrace(cpu_trace_.get());
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      cpu_trace_.transferCpuTrace(end_time);
    }

    auto trace = torch::profiler::impl::kineto::stopTrace();
    TORCH_CHECK(trace || !torch::profiler::kKinetoAvailable);
    addTraceEvents(trace);
    return trace;
  }

  void materializeOpEvents() {
    std::lock_guard<std::mutex> guard(state_mutex_);

    for (const auto& e : memory_events_) {
        cpu_trace_.addMemoryUsageActivity(
            kMemoryEventName,
            e.kineto_info,
            e.start_time,
            c10::Device(e.device_type, e.device_index),
            e.ptr,
            e.alloc_size,
            e.total_allocated,
            e.total_reserved);

      kineto_events_.emplace_back();
      auto& evt = kineto_events_.back();
      evt.name(kMemoryEventName)
          .startUs(e.start_time)
          .deviceIndex(e.device_index)
          .deviceType(e.device_type)
          .nBytes(e.alloc_size)
          .startThreadId(e.threadID);
    }

    for (const auto& e : op_events_) {
      if (e.end_us_ < e.start_us_) {
        // We initialize end_us_ to the smallest int64_t, so this means that
        // the op did not finish before we stopped profiling.
        continue;
      }

      cpu_trace_.addCPUActivity(
          e.name_,
          e.kineto_info_,
          e.correlation_id_,
          e.start_us_,
          e.end_us_);

      kineto_events_.emplace_back();
      kineto_events_.back()
          .name(e.name_)
          .startUs(e.start_us_)
          .durationUs(e.end_us_ - e.start_us_)
          .correlationId(e.correlation_id_)
          .deviceType(c10::DeviceType::CPU)
          .startThreadId(e.start_thread_id_)
          .endThreadId(e.end_thread_id_)
          .sequenceNr(e.sequence_number_)
          .fwdThreadId(e.forward_thread_id_)
          .scope(e.record_function_scope_)
          .setAsync(e.is_async_)
          .debugHandle(e.debug_handle_);

      if (!e.shapes_.empty()) {
        kineto_events_.back().shapes(e.shapes_);
      }

      if (!e.dtypes_.empty()) {
        kineto_events_.back().dtypes(e.dtypes_);
      }

      if (!e.stack_.empty()) {
        kineto_events_.back().stack(e.stack_);
      }

      if (e.module_hierarchy_) {
        kineto_events_.back().moduleHierarchy(*e.module_hierarchy_);
      }

      if (!e.extra_args_.empty()) {
        kineto_events_.back().flops(
            computeFlops(std::string(e.name_), e.extra_args_));
      }
      if (e.backend_) {
        kineto_events_.back().backend(*e.backend_);
      }
      kineto_events_.back().cuda_event_start_ = e.cuda_event_start_;
      kineto_events_.back().cuda_event_end_ = e.cuda_event_end_;
    }
    op_events_.clear();
  }

  void finalizeCPUTrace(std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace) {
#ifndef USE_KINETO
  }
#else // USE_KINETO
    TORCH_INTERNAL_ASSERT(
        cpu_trace->activities.size() == kineto_events_.size());
    // startThreadId_seqNum to pointer of activity.
    // Low-16bits of startThreadId and low-48bits seqNum are concatenated into
    // one uint64_t variable as key.
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>
        tidSeq2activity;
    uint64_t fwd_bwd_link_id = 1;

    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      if (kineto_event.hasShapes()) {
        activity.addMetadata("Input Dims", torch::profiler::impl::shapesToStr(kineto_event.shapes()));
      }
      if (kineto_event.hasStack()) {
        // NB: This is only for the JIT stack. The python stack (if applicable)
        //     is constructed later.
        activity.addMetadata(
            "Call stack", torch::profiler::impl::stacksToStr(kineto_event.stack(), ";"));
      }
      if (kineto_event.hasModuleHierarchy()) {
        activity.addMetadata(
            "Module Hierarchy",
            torch::profiler::impl::stacksToStr(kineto_event.moduleHierarchy(), "."));
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
            "Fwd thread id", std::to_string(kineto_event.fwdThreadId()));
        activity.addMetadata(
            "Sequence number", std::to_string(kineto_event.sequenceNr()));

        // From the time being, we need disable the forward/backward correlation feature to
        // workaround the crash bug.
        // TODO: by Mike Guo
        // reenable the forward/backward correlation when kineto fix the following raw pointer
        //    GenericTraceActivity.flow.linkedActivity
        // generateForwardBackwardLink(
        //     kineto_event, fwd_bwd_link_id, activity, tidSeq2activity);
      }
    }

    addPythonEvents(cpu_trace);
  }

  void addPythonEvents(std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace) {
    if (!tracePython()) {
      return;
    }

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
    ska::flat_hash_map<python_tracer::PyTraceEvent*, std::string>
        py_event_indices_{
            { nullptr,
              std::string("null") }};
    for (const auto i : c10::irange(py_events.size())) {
      py_event_indices_.insert({py_events[i].get(), std::to_string(i)});
    }

    ska::flat_hash_map<std::string, size_t> module_counter_;
    ska::flat_hash_map<size_t, std::string> module_id_map_;
    auto record_module_id = [&](python_tracer::PyTraceEvent* e) {
      if (e->call_type_ == python_tracer::CallType::kPyModuleCall &&
          module_id_map_.find(e->module_id_) == module_id_map_.end()) {
        // We use the fact that operator[] will default initialize new keys.
        module_id_map_[e->module_id_] =
            std::to_string(module_counter_[e->name_]++);
      }
    };

    // Python events
    std::vector<python_tracer::Replay> py_replay;
    for (const auto& e : py_events) {
      py_replay.push_back({e.get(), true});
      py_replay.push_back({e.get(), false});
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

    std::vector<libkineto::GenericTraceActivity> py_activities;
    auto py_events_it = py_events.begin();
    auto py_device = libkineto::processId();
    auto main_thread = libkineto::systemThreadId();
    auto push_py_event = [&]() {
      auto e = (*py_events_it).get();
      libkineto::GenericTraceActivity op(
          cpu_trace->span, libkineto::ActivityType::PYTHON_FUNCTION, e->name_);

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

      py_activities.push_back(op);
      py_events_it++;
    };

    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == kineto_events_.size());
    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& activity = cpu_trace->activities[idx];

      // Add any python events that occurred between this Kineto event and the
      // previous Kineto event.
      while (py_events_it != py_events.end() &&
             (*py_events_it)->endTime_ <= activity.endTime) {
        push_py_event();
      }

      auto python_caller = op_py_map.at(activity.startTime);
      activity.addMetadata(
          "python_caller_id", py_event_indices_.at(python_caller));

      // If the kineto event has a stack that means the JIT model has a stack
      // associated with it that we need to respect.
      if (!kineto_events_[idx].hasStack()) {
        std::vector<std::string> py_names;
        _push_reverse_order(python_caller, py_names);
        kineto_events_[idx].stack(py_names);
        activity.addMetadata("Call stack", torch::profiler::impl::stacksToStr(py_names, ";"));
      }
    }

    // Add any Python events which finish after the last Kineto event.
    while (py_events_it != py_events.end()) {
      push_py_event();
    }

    cpu_trace->activities.insert(cpu_trace->activities.end(), py_activities.begin(), py_activities.end());
  }

  void generateForwardBackwardLink(
      const KinetoEvent& kineto_event,
      uint64_t& fwd_bwd_link_id,
      libkineto::GenericTraceActivity& activity,
      std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>&
          tidSeq2activity) {
    if (kineto_event.fwdThreadId() > 0) {
      // act is backward op.
      uint64_t key = getForwardThreadKey(
          kineto_event.fwdThreadId(), kineto_event.sequenceNr());
      auto iter = tidSeq2activity.find(key);
      if (iter != tidSeq2activity.end()) {
        libkineto::GenericTraceActivity* fwd = iter->second;
#ifdef USE_KINETO_UPDATED
        fwd->flow.start = true;
#else
        activity.flow.linkedActivity = fwd; // Only destination side set this,
                                            // to distinguish with start side.
#endif
        activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
        activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
        ++fwd_bwd_link_id;
      }
    } else if (kineto_event.startThreadId() != 0) {
      // act is forward op.
      uint64_t key = getForwardThreadKey(
          kineto_event.startThreadId(), kineto_event.sequenceNr());
      // Assumption: Among all ops with same sequence number,
      // the one with biggest start time is most likely launching backward op.
      auto iter = tidSeq2activity.find(key);
      if (iter == tidSeq2activity.end()) {
        tidSeq2activity[key] = &activity;
      } else {
        // Now the sequence number is only incremented on creating a "Node"
        // object for backward pass, by calling
        // "at::sequence_number::get_and_increment()". Among all ops with same
        // sequence number, the one with biggest startTime is the one launching
        // backward op.
        if (activity.startTime >= iter->second->startTime) {
          tidSeq2activity[key] = &activity;
        }
      }
    }
  }
#endif // USE_KINETO

  void addTraceEvents(torch::profiler::impl::kineto::ActivityTraceWrapper& trace) {
#ifdef USE_KINETO
    const auto& events = *(trace.get()->activities());
    for (const auto& ev_ptr : events) {
      const auto& activity = *ev_ptr;
      // These events are already processed
      if (activity.type() != libkineto::ActivityType::CPU_OP &&
          activity.type() != libkineto::ActivityType::CPU_INSTANT_EVENT &&
          activity.type() != libkineto::ActivityType::USER_ANNOTATION &&
          activity.type() != libkineto::ActivityType::PYTHON_FUNCTION) {
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
#endif // USE_KINETO
  }

  uint64_t start_time_;
  std::set<torch::profiler::impl::ActivityType> activities_;
  std::deque<OpEventData> op_events_;
  std::deque<MemoryEventData> memory_events_;
  torch::profiler::impl::kineto::TraceWrapper cpu_trace_;
  std::vector<KinetoEvent> kineto_events_;
  // Optional, if event post-processing is enabled.
  std::function<void(std::vector<KinetoEvent>&)> event_post_process_cb_;
};

void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto registration_state_ptr = KinetoThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            auto state_ptr = KinetoThreadLocalState::getTLS();
            if (!state_ptr) {
              return nullptr;
            }
            const auto& config = state_ptr->config();
            auto corr_id = next_correlation_id();
            torch::profiler::impl::kineto::pushCorrelationId(corr_id);

            auto ctx_ptr = state_ptr->newOpEvent();
            auto data_ptr = ctx_ptr->data_;

            data_ptr->end_us_ = std::numeric_limits<int64_t>::min();
            data_ptr->correlation_id_ = corr_id;
            data_ptr->start_thread_id_ = fn.threadId();
            data_ptr->sequence_number_ = fn.seqNr();
            data_ptr->forward_thread_id_ = fn.forwardThreadId();
            data_ptr->record_function_scope_ = (uint8_t)fn.scope();
            data_ptr->is_async_ = fn.isAsync();
            data_ptr->debug_handle_ = fn.debugHandle();
            data_ptr->kineto_info_ = torch::profiler::impl::kineto::kineto_ids();
            data_ptr->name_ = fn.name();
            if (config.report_input_shapes) {
              data_ptr->shapes_ = torch::profiler::impl::inputSizes(fn);
              data_ptr->dtypes_ = torch::profiler::impl::inputTypes(fn);
            }
#if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
            // backward nodes source range corresponds to the forward node
            // TODO: consider using C++ stack trace
            if (config.with_stack &&
                fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
              auto cs = torch::profiler::impl::prepareCallstack(jit::currentCallstack());
              data_ptr->stack_ = callstackStr(cs);
            }
            if (config.with_modules &&
                fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
              data_ptr->module_hierarchy_ = jit::currentModuleHierarchy();
            }
#endif
            if (config.with_flops) {
              data_ptr->extra_args_ = torch::profiler::impl::saveExtraArgs(fn);
            }
            data_ptr->start_us_ = getTimeUs();

            if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
              try {
                torch::profiler::impl::cudaStubs()->record(
                    nullptr, &data_ptr->cuda_event_start_, nullptr);
              } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to record CUDA event. " << e.what();
              }
            }
            return ctx_ptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            auto state_ptr = KinetoThreadLocalState::getTLS();
            if (!state_ptr) {
              return;
            }
            const auto& config = state_ptr->config();
            auto* kineto_ctx_ptr =
                static_cast<KinetoObserverContext*>(ctx_ptr);
            TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
            auto data_ptr = kineto_ctx_ptr->data_;
            data_ptr->end_us_ = getTimeUs();
            data_ptr->end_thread_id_ = at::RecordFunction::currentThreadId();

            if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
              try {
                torch::profiler::impl::cudaStubs()->record(
                    nullptr, &data_ptr->cuda_event_end_, nullptr);
              } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to record CUDA event. " << e.what();
              }
            }

            torch::profiler::impl::kineto::popCorrelationId();
            torch::profiler::impl::kineto::recordThreadInfo();
          })
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes));
  registration_state_ptr->setCallbackHandle(handle);
}

} // namespace

void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  auto state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }

  auto ctx_ptr = state_ptr->newOpEvent();
  auto data_ptr = ctx_ptr->data_;
  data_ptr->start_us_ = start_time_us;
  data_ptr->end_us_ = end_time_us;
  data_ptr->correlation_id_ = std::numeric_limits<uint64_t>::max();
  data_ptr->start_thread_id_ = at::RecordFunction::currentThreadId();
  data_ptr->end_thread_id_ = data_ptr->start_thread_id_;
  data_ptr->sequence_number_ = -1;
  data_ptr->forward_thread_id_ = data_ptr->start_thread_id_;
  data_ptr->record_function_scope_ = (uint8_t)scope;
  data_ptr->is_async_ = false;
  data_ptr->debug_handle_ = debug_handle;
  data_ptr->kineto_info_ = torch::profiler::impl::kineto::kineto_ids();
  data_ptr->name_ = event_name;
  data_ptr->backend_ = backend_name;

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */

  torch::profiler::impl::kineto::recordThreadInfo();
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
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasCUDA(), activities);
}

void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    std::function<void(std::vector<KinetoEvent>&)>&& cb,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      config.state != ProfilerState::NVTX,
      "NVTX does not support post processing callback.");
  enableProfiler(config, activities, scopes);
  auto state_ptr = KinetoThreadLocalState::getTLS();
  state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(!profilerEnabled(), "Profiler is already enabled on this thread");
  if (config.state == ProfilerState::NVTX) {
    torch::profiler::impl::pushNVTXCallbacks(config, scopes);
    return;
  }

  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK);
  TORCH_CHECK(
      !activities.empty(), "No activities specified for Kineto profiler");

  auto state = std::make_shared<KinetoThreadLocalState>(config, activities);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  if (state->tracePython()) {
    python_tracer::call(python_tracer::Command::kStartOne);
  }

  if (activities.count(ActivityType::CPU)) {
    pushProfilingCallbacks(scopes);
  }

  torch::profiler::impl::kineto::startTrace();
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use
  // DebugInfoGuard
  auto state =
      c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<ProfilerThreadLocalStateBase*>(state.get());
  const auto& config = state_ptr->config();
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::NVTX),
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  if (state_ptr->config().state == ProfilerState::NVTX) {
    return std::make_unique<ProfilerResult>();
  }

  auto kineto_state_ptr = static_cast<KinetoThreadLocalState*>(state_ptr);
  if (kineto_state_ptr->tracePython()) {
    python_tracer::call(python_tracer::Command::kStop);
  }

  auto trace = kineto_state_ptr->finalizeTrace();
  if (kineto_state_ptr->tracePython()) {
    python_tracer::call(python_tracer::Command::kClear);
  }

  return std::make_unique<ProfilerResult>(
      kineto_state_ptr->start_time_,
      std::move(kineto_state_ptr->kineto_events_),
      std::move(trace));
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

ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    torch::profiler::impl::kineto::ActivityTraceWrapper trace)
    : trace_start_us_(start_time),
      events_(std::move(events)),
      trace_(std::move(trace)) {}
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  trace_.save(path);
}

} // namespace profiler
} // namespace autograd
} // namespace torch
