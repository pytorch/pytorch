#include <cstring>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>
#include <c10/util/variant.h>

#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/standalone/itt_observer.h>
#include <torch/csrc/profiler/standalone/nvtx_observer.h>
#include <torch/csrc/profiler/util.h>

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
inline int64_t getTimeUs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return torch::profiler::impl::getTime() / 1000;
#endif // USE_KINETO
}

using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::dtypesToStr;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::ProfilerStateBase;
using torch::profiler::impl::PyExtraFieldsBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shapesToStr;
using torch::profiler::impl::stacksToStr;

struct MetadataBase {
  MetadataBase(const std::shared_ptr<Result>& result)
      : kineto_activity_{result->kineto_activity_} {
    if (c10::holds_alternative<ExtraFields<EventType::Kineto>>(
            result->extra_fields_)) {
      // In order to add metadata we have to downcast from
      // `libkineto::ITraceActivity` to `libkineto::GenericTraceActivity`. We
      // know that all activities provided by PyTorch are of the correct type,
      // however Kineto profilers can (and do) add events that inherit directly
      // from ITraceActivity. As a result, any Result which was constructed from
      // an event that Kineto provided is unsafe to cast.
      if (!(SOFT_ASSERT(!hasKinetoActivity()))) {
        result->kineto_activity_ = nullptr;
      }
      kineto_activity_ = result->kineto_activity_;
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    if (kineto_activity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(kineto_activity_, key, value);
    }
  }

  bool hasKinetoActivity() const {
    return kineto_activity_ != nullptr;
  }

 private:
  const torch::profiler::impl::kineto::activity_t* kineto_activity_{nullptr};
};

struct AddTensorboardFields : public MetadataBase {
  AddTensorboardFields(
      const std::shared_ptr<Result>& result,
      KinetoEvent& kineto_event)
      : MetadataBase(result) {
    result->visit(*this);
    const auto module_hierarchy = kineto_event.moduleHierarchy();
    addMetadata("Module Hierarchy", stacksToStr(module_hierarchy.vec(), "."));
    addMetadata("Call stack", stacksToStr(kineto_event.stack().vec(), ";"));

    result->visit_if_base<PyExtraFieldsBase>([&, this](const auto& i) -> void {
      this->addMetadata("Python id", std::to_string(i.id_));

      c10::optional<std::string> parent_id;
      std::shared_ptr<Result> parent = result->parent_.lock();
      while (parent && !parent_id.has_value()) {
        parent->visit_if_base<PyExtraFieldsBase>(
            [&](const auto& j) { parent_id = std::to_string(j.id_); });
        parent = parent->parent_.lock();
      }
      this->addMetadata("Python parent id", parent_id.value_or("null"));
    });
  }

  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    if (py_call.module_.has_value()) {
      addMetadata("Python module id", std::to_string(py_call.module_->id_));
    }
  }

  template <typename T>
  void operator()(const T&) {}
};

struct AddGenericMetadata : public MetadataBase {
  AddGenericMetadata(
      std::shared_ptr<Result>& result,
      const torch::profiler::impl::ProfilerConfig* config)
      : MetadataBase(result), config_(config) {
    result->visit(*this);
    if (config->experimental_config.verbose) {
      result->visit_if_base<PyExtraFieldsBase>(
          [&, this](const auto& i) -> void {
            this->addMetadata("Python thread", std::to_string(i.python_tid_));
          });
    }
  }

  void operator()(ExtraFields<EventType::TorchOp>& op_event) {
    auto& shapes = op_event.inputs_.shapes_;
    if (!shapes.empty()) {
      addMetadata("Input Dims", shapesToStr(shapes));
    }

    auto& dtypes = op_event.inputs_.dtypes_;
    if (!dtypes.empty()) {
      addMetadata("Input type", dtypesToStr(dtypes));
    }

    if (config_ && !config_->experimental_config.performance_events.empty()) {
      auto& event_names = config_->experimental_config.performance_events;
      for (auto i = 0; i < op_event.perf_event_counters_->size(); ++i) {
        addMetadata(
            event_names[i],
            std::to_string((*op_event.perf_event_counters_)[i]));
      }
    }

    // add information about an associated forward op, if a sequence number
    // is available (e.g. during training)
    if (op_event.sequence_number_ >= 0) {
      addMetadata("Fwd thread id", std::to_string(op_event.forward_tid_));
      addMetadata("Sequence number", std::to_string(op_event.sequence_number_));
    }
  }

  void operator()(ExtraFields<EventType::Backend>& backend_event) {
    if (!backend_event.backend_.empty()) {
      addMetadata("Backend", "\"" + backend_event.backend_ + "\"");
    }
  }

  void operator()(const ExtraFields<EventType::Allocation>& alloc) {
    addMetadata("Device Type", std::to_string((int8_t)alloc.device_type_));
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    addMetadata("Addr", std::to_string(reinterpret_cast<intptr_t>(alloc.ptr_)));
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    if (alloc.total_allocated_ >= 0) {
      addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    }
    if (alloc.total_reserved_ >= 0) {
      addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
    }
  }

  void operator()(const ExtraFields<EventType::OutOfMemory>& alloc) {
    addMetadata("Device Type", std::to_string((int8_t)alloc.device_type_));
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    if (alloc.total_allocated_ >= 0) {
      addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    }
    if (alloc.total_reserved_ >= 0) {
      addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
    }
  }

  template <typename T>
  void operator()(const T&) {}

 private:
  /* To get names of the performance events */
  const torch::profiler::impl::ProfilerConfig* config_;
};

// Assumption: Total threads number will not exceed 2^16-1, and total ops will
// not exceed 2^48 -1.
static inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

struct KinetoThreadLocalState : public ProfilerStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerStateBase(config),
        start_time_(getTimeUs()),
        record_queue_(config, activities) {}
  ~KinetoThreadLocalState() override = default;

  static KinetoThreadLocalState* get(bool global) {
    auto* state = ProfilerStateBase::get(/*global=*/global);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        state == nullptr ||
        state->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(state);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      record_queue_.getSubqueue()->emplace_allocation_event(
          torch::profiler::impl::getApproximateTime(),
          ptr,
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  void reportOutOfMemory(
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      record_queue_.getSubqueue()->emplace_ooms_event(
          torch::profiler::impl::getApproximateTime(),
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  const post_process_t& getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(post_process_t&& cb) {
    event_post_process_cb_ = std::move(cb);
  }

  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>
  finalizeTrace() {
    auto end_time = getTimeUs();
    record_queue_.stop();

    std::lock_guard<std::mutex> guard(state_mutex_);
    auto converter = clock_converter_.makeConverter();
    auto records_and_trace =
        record_queue_.getRecords(converter, start_time_, end_time);

    materializeOpEvents(records_and_trace.first);

    // finalizeCPUTrace(cpu_trace_.get());

    // `kineto_events_` does not include Python events. Instead it exposes them
    // via the `stacks` property.
    kineto_events_.erase(
        std::remove_if(
            kineto_events_.begin(),
            kineto_events_.end(),
            [](const auto& i) { return i.isPythonFunction(); }),
        kineto_events_.end());

    return std::move(records_and_trace.second);
  }

  template <typename T>
  void invokeCallback(T& t) {
    if (event_post_process_cb_) {
      event_post_process_cb_(t.debug_handle_, t.jit_stack_, t.jit_modules_);
    }
  }

  void materializeOpEvents(std::vector<std::shared_ptr<Result>>& events) {
    for (auto& e : events) {
      if (e->parent_.expired()) {
        event_tree_.push_back(e);
      }

      if (e->finished_) {
        e->visit(c10::overloaded(
            [this](ExtraFields<EventType::TorchOp>& i) { invokeCallback(i); },
            [this](ExtraFields<EventType::Backend>& i) { invokeCallback(i); },
            [](auto&) {}));

        kineto_events_.emplace_back(e, config_.experimental_config.verbose);
        AddTensorboardFields add_tb(e, kineto_events_.back());
        AddGenericMetadata add_generic(e, &config_);

        // It is not safe to use the activity after post processing.
        e->kineto_activity_ = nullptr;
      }
    }
  }

  void finalizeCPUTrace(
      std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace) {
#ifndef USE_KINETO
  }
#else // USE_KINETO
    TORCH_INTERNAL_ASSERT(
        cpu_trace->activities.size() == kineto_events_.size());
    // startThreadId_seqNum to pointer of activity.
    // Low-16bits of startThreadId and low-48bits seqNum are concatenated into
    // one uint64_t variable as key.

    // From the time being, we need disable the forward/backward correlation
    // feature to workaround the crash bug.
    // TODO: by Mike Guo
    // reenable the forward/backward correlation when kineto fix the following
    // raw pointer
    //    GenericTraceActivity.flow.linkedActivity
    /*
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>
        tidSeq2activity;

    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      // add information about an associated forward op, if a sequence number
      // is available (e.g. during training)
      if (kineto_event.sequenceNr() >= 0) {
        generateForwardBackwardLink(
            kineto_event, fwd_bwd_link_id, activity, tidSeq2activity);
      }
    }
    */
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
        fwd->flow.start = true;
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

  uint64_t start_time_;
  torch::profiler::impl::ApproximateClockToUnixTimeConverter clock_converter_;
  torch::profiler::impl::RecordQueue record_queue_;
  std::vector<KinetoEvent> kineto_events_;
  std::vector<experimental_event_t> event_tree_;
  // Optional, if event post-processing is enabled.
  post_process_t event_post_process_cb_;
};

template <bool use_global_state_ptr = false>
std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  auto state_ptr = KinetoThreadLocalState::get(use_global_state_ptr);
  if (!state_ptr) {
    return nullptr;
  }
  return state_ptr->record_queue_.getSubqueue()->begin_op(fn);
}

// @lint-ignore CLANGTIDY clang-diagnostic-unused-parameter
template <bool use_global_state_ptr = false>
void onFunctionExit(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  auto state_ptr = KinetoThreadLocalState::get(use_global_state_ptr);
  if (!state_ptr) {
    return;
  }
  const auto& config = state_ptr->config();
  auto* kineto_ctx_ptr =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
  kineto_ctx_ptr->event_->end_time_ =
      torch::profiler::impl::getApproximateTime();
  if (!config.experimental_config.performance_events.empty()) {
    state_ptr->record_queue_.getSubqueue()->disable_perf_profiler(
        *kineto_ctx_ptr->event_->counters_);
  }
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();
  if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      auto fallback = kineto_ctx_ptr->fallback_;
      TORCH_INTERNAL_ASSERT(fallback != nullptr);
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &fallback->cuda_event_end_, nullptr);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  }

  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::popUserCorrelationId();
  } else {
    torch::profiler::impl::kineto::popCorrelationId();
  }
}

template <bool use_global_callback = false>
void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto registration_state_ptr =
      KinetoThreadLocalState::get(use_global_callback);
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto recordFunctionCallback =
      at::RecordFunctionCallback(
          onFunctionEnter<use_global_callback>,
          onFunctionExit<use_global_callback>)
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes);

  auto handle = c10::guts::if_constexpr<use_global_callback>(
      [&] { return at::addGlobalCallback(recordFunctionCallback); },
      [&] { return at::addThreadLocalCallback(recordFunctionCallback); });
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
  TORCH_INTERNAL_ASSERT(
      KinetoThreadLocalState::get(/*global=*/true) == nullptr,
      "On-demand profiling does not support post processing callback");

  auto state_ptr = KinetoThreadLocalState::get(/*global=*/false);
  if (!state_ptr) {
    return;
  }

  state_ptr->record_queue_.getSubqueue()->emplace_backend_event(
      start_time_us,
      end_time_us,
      debug_handle,
      scope,
      event_name,
      backend_name);

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::ITT) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasCUDA(), activities, config.experimental_config);

  if (config.experimental_config.performance_events.size()) {
    /* For now only CPU activity is supported */
    TORCH_CHECK(
        activities.count(torch::autograd::profiler::ActivityType::CPU),
        "Cannot run cpu hardware profiler without CPU activities, please only use CPU activity type");
    /*
     * Sending a warning and passing the non-standard event to the backend
     * Backend can abort if the event is not supported.
     * TODO Should we gracefully drop the invalid event if we have atleast one
     * valid?
     */
    auto is_standard_event = [](const std::string& event) -> bool {
      for (auto e : torch::profiler::ProfilerPerfEvents) {
        if (!std::strcmp(event.c_str(), e)) {
          return true;
        }
      }
      return false;
    };

    for (const auto& e : config.experimental_config.performance_events) {
      if (!is_standard_event(e)) {
        TORCH_WARN("Forwarding a non-standard CPU performance event : ", e);
      }
    }
  }
}

void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    post_process_t&& cb,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      config.state != ProfilerState::NVTX,
      "NVTX does not support post processing callback.");
  TORCH_CHECK(
      config.state != ProfilerState::ITT,
      "ITT does not support post processing callback.");
  TORCH_INTERNAL_ASSERT(
      KinetoThreadLocalState::get(/*global=*/true) == nullptr,
      "On-demand profiling does not support post processing callback");

  enableProfiler(config, activities, scopes);
  auto state_ptr = KinetoThreadLocalState::get(config.global());
  state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  const auto has_cpu = activities.count(ActivityType::CPU);
  TORCH_CHECK(
      KinetoThreadLocalState::get(/*global=*/config.global()) == nullptr,
      "Profiler is already enabled",
      (config.global() ? "." : " on this thread."));

  if (config.state == ProfilerState::NVTX) {
    torch::profiler::impl::pushNVTXCallbacks(config, scopes);
    return;
  } else if (config.state == ProfilerState::ITT) {
    torch::profiler::impl::pushITTCallbacks(config, scopes);
    return;
  }

  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK || config.global());
  TORCH_CHECK(!activities.empty(), "No activities specified.");
  TORCH_INTERNAL_ASSERT(
      has_cpu || !config.global(),
      "Ondemand profiling must enable CPU tracing");

  KinetoThreadLocalState::push(
      std::make_shared<KinetoThreadLocalState>(config, activities));

  if (has_cpu) {
    config.global() ? pushProfilingCallbacks</*global=*/true>(scopes)
                    : pushProfilingCallbacks</*global=*/false>(scopes);
  }

  if (!config.global()) {
    torch::profiler::impl::kineto::startTrace();
  }
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  auto state_ptr = ProfilerStateBase::pop();
  const auto& config = state_ptr->config();
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::KINETO_ONDEMAND ||
           config.state == ProfilerState::NVTX ||
           config.state == ProfilerState::ITT),
      "Can't disable Kineto profiler when it's not running");

  state_ptr->removeCallback();

  // Traces are converged via libkineto automatically for ondemand flow
  if (state_ptr->config().global()) {
    (void)std::static_pointer_cast<KinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  // Shared among NVTX, KINETO, KINETO_GPU_FALLBACK
  std::unique_ptr<ProfilerResult> result;
  if (state_ptr->config().state == ProfilerState::NVTX) {
    result = std::make_unique<ProfilerResult>();
  }

  if (config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    auto kineto_state_ptr =
        std::static_pointer_cast<KinetoThreadLocalState>(state_ptr);
    auto trace = kineto_state_ptr->finalizeTrace();
    result = std::make_unique<ProfilerResult>(
        kineto_state_ptr->start_time_,
        std::move(kineto_state_ptr->kineto_events_),
        std::move(trace),
        std::move(kineto_state_ptr->event_tree_));
  }

  return result;
}

KinetoEvent::KinetoEvent(
    std::shared_ptr<const torch::profiler::impl::Result> result,
    const bool verbose)
    : result_{result} {
  TORCH_INTERNAL_ASSERT(result != nullptr);

  if (verbose) {
    // Populate Python stack
    auto parent = result_->parent_.lock();
    while (parent != nullptr) {
      parent->visit_if_base<PyExtraFieldsBase>(
          [&](const auto& i) { python_stack_.push_back(parent->name()); });
      parent = parent->parent_.lock();
    }
  }
}

bool KinetoEvent::isPythonFunction() const {
  bool out{false};
  result_->visit_if_base<PyExtraFieldsBase>([&](const auto&) { out = true; });
  return out;
}

const c10::ArrayRef<std::string> KinetoEvent::stack() const {
  auto get = [&](const auto& i) -> auto& {
    return !i.jit_stack_.empty() ? i.jit_stack_ : python_stack_;
  };

  using out_t = const c10::ArrayRef<std::string>;
  return result_->visit(c10::overloaded(
      [&](const ExtraFields<EventType::TorchOp>& i) -> out_t { return get(i); },
      [&](const ExtraFields<EventType::Backend>& i) -> out_t { return get(i); },
      [&](const auto&) -> out_t { return python_stack_; }));
}

const c10::ArrayRef<std::string> KinetoEvent::moduleHierarchy() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& e)
          -> const c10::ArrayRef<std::string> { return e.jit_modules_; },
      [](const ExtraFields<EventType::Backend>& e)
          -> const c10::ArrayRef<std::string> { return e.jit_modules_; },
      [](const auto&) -> const c10::ArrayRef<std::string> { return {}; }));
}

uint64_t KinetoEvent::durationUs() const {
  return (result_->endTimeNS() - result_->start_time_ns_) / 1000;
}

int64_t KinetoEvent::debugHandle() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& i) { return i.debug_handle_; },
      [](const ExtraFields<EventType::Backend>& i) { return i.debug_handle_; },
      [](const auto&) -> int64_t { return -1; }));
}

uint8_t KinetoEvent::deviceIndex() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::Allocation>& i) {
        return static_cast<uint8_t>(i.device_index_);
      },
      [](const ExtraFields<EventType::OutOfMemory>& i) {
        return static_cast<uint8_t>(i.device_index_);
      },
      [&](const auto&) {
        return static_cast<uint8_t>(result_->kineto_info_.device);
      }));
}

bool KinetoEvent::hasStack() const {
  return !stack().empty();
}

int64_t KinetoEvent::cudaElapsedUs() const {
  auto cuda_event_start = fallbackStart();
  auto cuda_event_end = fallbackEnd();
  if (!cuda_event_start || !cuda_event_end) {
    return -1;
  }
  try {
    return (int64_t)torch::profiler::impl::cudaStubs()->elapsed(
        &cuda_event_start, &cuda_event_end);
  } catch (std::exception& e) {
    LOG(WARNING) << "Failed to measure time between two CUDA events. "
                 << e.what();
  }
  return -1;
}

void KinetoEvent::getPerfEventCounters(std::vector<uint64_t>& in) const {
  return result_->visit(c10::overloaded(
      [&in](const ExtraFields<EventType::TorchOp>& e) -> void {
        const size_t n = e.perf_event_counters_->size();
        // should be rare
        if (in.size() < n) {
          in.resize(n, 0);
        }
        for (size_t i = 0; i < n; ++i) {
          in[i] = (*e.perf_event_counters_)[i];
        }
      },
      [](const auto&) -> void { return; }));
}

#define FORWARD_FROM_RESULT(method_name, result_expr)                        \
  decltype(std::declval<KinetoEvent>().method_name())                        \
  KinetoEvent::method_name() const {                                         \
    return static_cast<decltype(std::declval<KinetoEvent>().method_name())>( \
        result_->result_expr);                                               \
  }

FORWARD_FROM_RESULT(startThreadId, start_tid_)
FORWARD_FROM_RESULT(endThreadId, endTID())
FORWARD_FROM_RESULT(activityType, kinetoType())
FORWARD_FROM_RESULT(name, name())
FORWARD_FROM_RESULT(deviceType, deviceType())
FORWARD_FROM_RESULT(startUs, start_time_ns_ / 1000)
FORWARD_FROM_RESULT(correlationId, correlationID())
FORWARD_FROM_RESULT(deviceResourceId, kineto_info_.resource)
#undef FORWARD_FROM_RESULT

// Most of the fields in `KinetoEvent` only make sense for a single event type.
// (Generally TorchOp.) For all other types they simply return the default
// value. This macro provides a succinct way of expressing this behavior.
#define TYPED_ATTR_WITH_DEFAULT(                                       \
    event_type, method_name, expression, default_value)                \
  decltype(std::declval<KinetoEvent>().method_name())                  \
  KinetoEvent::method_name() const {                                   \
    using out_t = decltype(std::declval<KinetoEvent>().method_name()); \
    return result_->visit(c10::overloaded(                             \
        [](const ExtraFields<EventType::event_type>& e) -> out_t {     \
          return expression;                                           \
        },                                                             \
        [](const auto&) -> out_t { return default_value; }));          \
  }

#define TYPED_ATTR(event_type, method_name, expression) \
  TYPED_ATTR_WITH_DEFAULT(event_type, method_name, expression, {})

TYPED_ATTR_WITH_DEFAULT(TorchOp, sequenceNr, e.sequence_number_, -1)
TYPED_ATTR(TorchOp, fwdThreadId, e.sequence_number_ >= 0 ? e.forward_tid_ : 0)
TYPED_ATTR(TorchOp, hasShapes, !e.inputs_.shapes_.empty())
TYPED_ATTR(TorchOp, shapes, e.inputs_.shapes_)
TYPED_ATTR(TorchOp, hasTypes, !e.inputs_.dtypes_.empty())
TYPED_ATTR(TorchOp, dtypes, e.inputs_.dtypes_)
TYPED_ATTR(TorchOp, scope, static_cast<uint8_t>(e.scope_))
TYPED_ATTR(TorchOp, hasModuleHierarchy, !e.jit_modules_.empty())
TYPED_ATTR(TorchOp, isAsync, e.is_async_)
TYPED_ATTR(TorchOp, fallbackStart, e.gpu_fallback_.cuda_event_start_)
TYPED_ATTR(TorchOp, fallbackEnd, e.gpu_fallback_.cuda_event_end_)
TYPED_ATTR(
    TorchOp,
    flops,
    !e.extra_args_.empty() ? computeFlops(e.name_, e.extra_args_) : 0)
TYPED_ATTR(Backend, backend, e.backend_)
TYPED_ATTR(Allocation, nBytes, e.alloc_size_)
TYPED_ATTR(Kineto, linkedCorrelationId, [&]() {
  const auto linked = e.linked_activity_.lock();
  return linked ? linked->correlationID() : 0;
}())
#undef TYPED_ATTR
#undef TYPED_ATTR_WITH_DEFAULT

ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&&
        trace,
    std::vector<experimental_event_t>&& event_tree)
    : trace_start_us_(start_time),
      events_(std::move(events)),
      trace_(std::move(trace)),
      event_tree_(std::move(event_tree)) {}
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  trace_->save(path);
}

} // namespace profiler
} // namespace autograd
} // namespace torch
