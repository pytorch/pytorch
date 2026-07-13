#include <fmt/format.h>
#include <cstring>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>
#include <fmt/format.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/standalone/itt_observer.h>
#include <torch/csrc/profiler/standalone/nvtx_observer.h>
#include <torch/csrc/profiler/standalone/privateuse1_observer.h>
#include <torch/csrc/profiler/util.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifdef USE_KINETO
#include <ApproximateClock.h>
#include <libkineto.h>
#include <time_since_epoch.h>
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type();
__attribute__((weak)) int acc_get_device_type() {
  TORCH_CHECK(
      false,
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch {
namespace autograd::profiler {

namespace {
inline int64_t getTimeNs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return c10::getTime();
#endif // USE_KINETO
}

using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::get_record_concrete_inputs_enabled;
using torch::profiler::impl::ivalueListToStr;
using torch::profiler::impl::ivalueToStr;
using torch::profiler::impl::op_input_t;
using torch::profiler::impl::ProfilerStateBase;
using torch::profiler::impl::PyExtraFieldsBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shape;
using torch::profiler::impl::shapesToStr;
using torch::profiler::impl::stacksToStr;
using torch::profiler::impl::strListToStr;
using torch::profiler::impl::TensorMetadata;
using torch::profiler::impl::variantShapesToStr;

// Helper function to check if ProfilerState is a Kineto-compatible state
inline bool isKinetoCompatibleState(ProfilerState state) {
  return state == ProfilerState::KINETO ||
      state == ProfilerState::KINETO_GPU_FALLBACK ||
      state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK ||
      state == ProfilerState::KINETO_PRIVATEUSE1;
}

// Helper function to check if ProfilerState is valid for disabling profiler
inline bool isValidDisableState(ProfilerState state) {
  return isKinetoCompatibleState(state) ||
      state == ProfilerState::KINETO_ONDEMAND || state == ProfilerState::NVTX ||
      state == ProfilerState::ITT || state == ProfilerState::PRIVATEUSE1;
}

// Helper function to check if ProfilerState uses an external tracer
// (NVTX/ITT/PRIVATEUSE1 - these use their own tracing callbacks, not Kineto)
inline bool isExternalTracerState(ProfilerState state) {
  return state == ProfilerState::NVTX || state == ProfilerState::ITT ||
      state == ProfilerState::PRIVATEUSE1;
}

inline bool hasRequestedDeviceActivity(
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  return activities.contains(ActivityType::CUDA) ||
      activities.contains(ActivityType::XPU) ||
      activities.contains(ActivityType::MTIA) ||
      activities.contains(ActivityType::HPU) ||
      activities.contains(ActivityType::PrivateUse1);
}

struct OpArgData {
  bool hasData;
  std::vector<shape> shapes;
  std::vector<std::string> dtypes;
  std::vector<c10::IValue> concreteInputs;
  std::vector<std::vector<int64_t>> shapesForKinetoEvent;
  std::vector<shape> strides;
};

auto parseArgData(
    const std::vector<op_input_t>& input_shapes,
    const std::vector<op_input_t>& concreteInputs) {
  if (input_shapes.empty()) {
    return OpArgData{.hasData = false};
  }

  std::vector<shape> shapes(input_shapes.size());
  std::vector<shape> strides(input_shapes.size());
  std::vector<std::vector<int64_t>> shapesForKinetoEvent(input_shapes.size());

  std::vector<std::string> dtypes(input_shapes.size());
  std::vector<c10::IValue> concrete_inputs_list;

  for (const auto& i : c10::irange(input_shapes.size())) {
    std::visit(
        c10::overloaded(
            [&](const TensorMetadata& t) {
              shapes[i] = t.sizes_;
              shapesForKinetoEvent[i] = t.sizes_;
              dtypes[i] = std::string(scalarTypeToTypeMeta(t.dtype_).name());
              strides[i] = t.strides_;
            },
            [&](const std::vector<TensorMetadata>& l) {
              std::vector<std::vector<int64_t>> shape;
              shape.reserve(l.size());
              std::vector<std::vector<int64_t>> stride;
              stride.reserve(l.size());
              for (const auto& t : l) {
                shape.emplace_back(t.sizes_);
                stride.emplace_back(t.strides_);
              }
              shapes[i] = shape;
              strides[i] = stride;
              dtypes[i] = "TensorList";
            },
            [&](const c10::IValue&) { dtypes[i] = "Scalar"; },
            [&](const auto&) {}),
        input_shapes[i]);
  }

  // If we recorded concrete inputs, then parse them
  if (input_shapes.size() == concreteInputs.size() && !concreteInputs.empty()) {
    concrete_inputs_list.resize(input_shapes.size());

    for (const auto& i : c10::irange(input_shapes.size())) {
      std::visit(
          c10::overloaded(
              [&](const c10::IValue& val) { concrete_inputs_list[i] = val; },
              [&](const auto&) {}),
          input_shapes[i]);
      std::visit(
          c10::overloaded(
              [&](const c10::IValue& val) {
                concrete_inputs_list[i] = val;
                dtypes[i] = "ScalarList";
              },
              [&](const auto&) {}),
          concreteInputs[i]);
    }
  }

  return OpArgData{
      .hasData = true,
      .shapes = shapes,
      .dtypes = dtypes,
      .concreteInputs = concrete_inputs_list,
      .shapesForKinetoEvent = shapesForKinetoEvent,
      .strides = strides};
}

struct MetadataBase {
  /* implicit */ MetadataBase(const std::shared_ptr<Result>& result)
      : kinetoActivity_{result->kineto_activity_} {
    if (std::holds_alternative<ExtraFields<EventType::Kineto>>(
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
      kinetoActivity_ = result->kineto_activity_;
    }
  }

  void addMetadata(
      const std::string& key,
      const std::string& value,
      bool quote = false) {
    if (kinetoActivity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<torch::profiler::impl::kineto::activity_t*>(
              kinetoActivity_),
          key,
          value,
          quote);
    }
  }

  bool hasKinetoActivity() const {
    return kinetoActivity_ != nullptr;
  }

 private:
  const torch::profiler::impl::kineto::activity_t* kinetoActivity_{nullptr};
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

      std::optional<std::string> parent_id;
      std::shared_ptr<Result> parent = result->parent_.lock();
      while (parent && !parent_id.has_value()) {
        parent->visit_if_base<PyExtraFieldsBase>(
            [&](const auto& j) { parent_id = std::to_string(j.id_); });
        parent = parent->parent_.lock();
      }
      this->addMetadata("Python parent id", parent_id.value_or("null"));
      if (i.caller_.line_no_ > 0) {
        this->addMetadata(
            "CallFrom",
            fmt::format("{}:{}", i.caller_.filename_.str(), i.caller_.line_no_),
            /*quote=*/true);
      }
    });
  }

  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    if (py_call.module_.has_value()) {
      addMetadata("Python module id", std::to_string(py_call.module_->id_));
    }
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}
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
    const auto arg_data =
        parseArgData(op_event.inputs_, op_event.concrete_inputs_);

    if (arg_data.hasData) {
      if (get_record_concrete_inputs_enabled()) {
        addMetadata("Input Dims", variantShapesToStr(arg_data.shapes));
      } else {
        addMetadata("Input Dims", shapesToStr(arg_data.shapesForKinetoEvent));
      }
      addMetadata("Input Strides", variantShapesToStr(arg_data.strides));
      addMetadata("Input type", strListToStr(arg_data.dtypes));
      if (!arg_data.concreteInputs.empty()) {
        addMetadata(
            "Concrete Inputs", ivalueListToStr(arg_data.concreteInputs));
      }
    }

    // Add metadata for kwinputs if exist
    for (const auto& [key, val] : op_event.kwinputs_) {
      if (key == "stream" && !val.isInt()) {
        LOG(WARNING) << "Inputted stream is not an int for op: "
                     << op_event.name_ << " skipping";
        continue;
      }

      // Until needed, let's limit the kwargs to only ints, doubles, strings,
      // bools, and list of strings
      bool isValidType =
          val.isInt() || val.isDouble() || val.isString() || val.isBool();
      bool isStringList = false;

      if (!isValidType && val.isList()) {
        // Check if it's a list of strings
        auto list = val.toListRef();
        isStringList = std::ranges::all_of(
            list, [](const c10::IValue& item) { return item.isString(); });
      }

      if (!isValidType && !isStringList) {
        LOG(WARNING)
            << "Inputted kwarg: " << key
            << " is not an int, double, string, bool, or list of strings for op: "
            << op_event.name_ << " skipping";
        continue;
      }

      if (isStringList) {
        // For list of strings, use ivalueListToStr
        auto list = val.toListRef();
        std::vector<c10::IValue> stringList(list.begin(), list.end());
        addMetadata(key, ivalueListToStr(stringList));
      } else {
        bool isString = val.isString();
        addMetadata(key, ivalueToStr(val, isString));
      }
    }
    // Add extra metadata if any
    for (const auto& [key, val] : op_event.extra_meta_) {
      addMetadata(key, val);
    }

    if (config_ && !config_->experimental_config.performance_events.empty()) {
      auto& event_names = config_->experimental_config.performance_events;
      for (const auto i : c10::irange(op_event.perf_event_counters_->size())) {
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
    addMetadata(
        "Record function id", std::to_string(op_event.record_function_id_));
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
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
  }

  void operator()(const ExtraFields<EventType::OutOfMemory>& alloc) {
    addMetadata("Device Type", std::to_string((int8_t)alloc.device_type_));
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}

 private:
  /* To get names of the performance events */
  const torch::profiler::impl::ProfilerConfig* config_;
};

// Lightweight metadata pass for trace_only mode: annotates Kineto activities
// with the same metadata as materializeOpEvents but without creating
// KinetoEvent wrappers or building eventTree.
void addTraceMetadata(
    std::vector<std::shared_ptr<Result>>& events,
    const torch::profiler::impl::ProfilerConfig& config,
    int64_t trace_end_ns) {
  for (auto& e : events) {
    // Unfinished events automatically have end time set to trace end time
    if (!e->finished_) {
      e->visit(c10::overloaded(
          [trace_end_ns](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ = trace_end_ns;
          },
          [](auto&) {}));
    }

    if (!e->kineto_activity_) {
      continue;
    }
    AddGenericMetadata add_generic(e, &config);

    // Subset of AddTensorboardFields that doesn't require KinetoEvent or
    // parent chain (no python_stack_, no Python parent id).
    MetadataBase tb(e);
    e->visit(c10::overloaded(
        [&](const ExtraFields<EventType::TorchOp>& i) {
          tb.addMetadata("Module Hierarchy", stacksToStr(i.jit_modules_, "."));
          tb.addMetadata("Call stack", stacksToStr(i.jit_stack_, ";"));
        },
        [&](const ExtraFields<EventType::Backend>& i) {
          tb.addMetadata("Module Hierarchy", stacksToStr(i.jit_modules_, "."));
          tb.addMetadata("Call stack", stacksToStr(i.jit_stack_, ";"));
        },
        [](const auto&) {}));
    e->visit_if_base<PyExtraFieldsBase>([&](const auto& i) {
      tb.addMetadata("Python id", std::to_string(i.id_));
    });
    e->visit(c10::overloaded(
        [&](const ExtraFields<EventType::PyCall>& py_call) {
          if (py_call.module_.has_value()) {
            tb.addMetadata(
                "Python module id", std::to_string(py_call.module_->id_));
          }
        },
        [](const auto&) {}));

    e->kineto_activity_ = nullptr;
  }
}

struct KinetoThreadLocalState : public ProfilerStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerStateBase(config),
        startTime(getTimeNs()),
        recordQueue(config, std::move(activities)) {}
  ~KinetoThreadLocalState() override = default;

  static std::shared_ptr<KinetoThreadLocalState> getGlobal() {
    std::shared_ptr<ProfilerStateBase> state = ProfilerStateBase::getGlobal();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        state == nullptr ||
        state->profilerType() == ActiveProfilerType::KINETO);
    return std::static_pointer_cast<KinetoThreadLocalState>(std::move(state));
  }

  static KinetoThreadLocalState* getTLS() {
    ProfilerStateBase* state = ProfilerStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        state == nullptr ||
        state->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(state);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  void reportVulkanEventToProfiler(torch::profiler::impl::vulkan_id_t id) {
    if (!config_.disabled()) {
      recordQueue.getSubqueue()->emplace_vulkan_event(
          c10::getApproximateTime(), id);
    }
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      recordQueue.getSubqueue()->emplace_allocation_event(
          c10::getApproximateTime(),
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
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      recordQueue.getSubqueue()->emplace_ooms_event(
          c10::getApproximateTime(),
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  void setEventPostProcessingCallback(post_process_t&& cb) {
    eventPostProcessCb = std::move(cb);
  }

  void pausePython() {
    recordQueue.stop();
  }

  void resumePython() {
    recordQueue.restart();
  }

  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>
  finalizeTrace() {
    auto end_time = getTimeNs();
    recordQueue.stop();

    std::lock_guard<std::mutex> guard(state_mutex_);
    auto converter = clockConverter.makeConverter();
#ifdef USE_KINETO
    libkineto::get_time_converter() = converter;
#endif
    auto records_and_trace =
        recordQueue.getRecords(std::move(converter), startTime, end_time);

    if (config_.experimental_config.trace_only) {
      addTraceMetadata(records_and_trace.first, config_, end_time);
    } else {
      materializeOpEvents(records_and_trace.first, end_time);
    }

    return std::move(records_and_trace.second);
  }

  template <typename T>
  void invokeCallback(T& t) {
    if (eventPostProcessCb) {
      eventPostProcessCb(t.debug_handle_, t.jit_stack_, t.jit_modules_);
    }
  }

  void materializeOpEvents(
      std::vector<std::shared_ptr<Result>>& events,
      int64_t trace_end_ns) {
    for (auto& e : events) {
      if (e->parent_.expired() && e->deviceType() == c10::DeviceType::CPU) {
        eventTree.push_back(e);
      }

      // Unfinished events automatically have end time set to trace end time
      if (!e->finished_) {
        e->visit(c10::overloaded(
            [trace_end_ns](ExtraFields<EventType::TorchOp>& i) {
              i.end_time_ns_ = trace_end_ns;
            },
            [](auto&) {}));
      }

      e->visit(c10::overloaded(
          [this](ExtraFields<EventType::TorchOp>& i) { invokeCallback(i); },
          [this](ExtraFields<EventType::Backend>& i) { invokeCallback(i); },
          [](auto&) {}));

      kinetoEvents.emplace_back(e, config_.experimental_config.verbose);
      AddTensorboardFields add_tb(e, kinetoEvents.back());
      AddGenericMetadata add_generic(e, &config_);

      // It is not safe to use the activity after post processing.
      e->kineto_activity_ = nullptr;
    }
  }

  uint64_t startTime;
  c10::ApproximateClockToUnixTimeConverter clockConverter;
  torch::profiler::impl::RecordQueue recordQueue;
  std::vector<KinetoEvent> kinetoEvents;
  std::vector<experimental_event_t> eventTree;
  // Optional, if event post-processing is enabled.
  post_process_t eventPostProcessCb;
};

// Coordinates one global RecordFunction callback session (KINETO_ONDEMAND, or
// KINETO with profile_all_threads) against teardown. Those callbacks fire on
// every thread and touch the profiler state while disableProfiler() finalizes
// and frees it on another thread.
//
// While the session is active, each callback invocation brackets only its own
// short critical section - getGlobal() plus the RecordQueue access - with
// enter()/exit(). Teardown calls drain(), which closes the session to new
// callbacks and blocks until the in-flight count reaches zero, so finalize
// never races a live critical section.
//
// The enter()/isActive() re-check is a handshake: a callback that joined before
// teardown is always waited for, while one arriving after bails without
// touching the state.
//
// Bracketing only the critical section (not the op body) is what keeps drain()
// from deadlocking against the GIL, but it lets an op enter under one session
// and exit after that session is torn down (its RecordQueue and event freed). A
// per-session generation (see generation()), stamped on each callback's
// ObserverContext, closes that gap: onFunctionExitGlobal drops a straddling
// exit on generation mismatch instead of dereferencing the freed event.
//
// The drain counter is not a std::latch: a std::latch fixes its count at
// construction, cannot be incremented afterward, and is single-use. Here the
// number of in-flight callbacks is unknown up front, grows dynamically as ops
// dispatch, and the session must be re-armed for every enable/disable cycle,
// none of which std::latch supports. The counter stays lock-free on the per-op
// hot path; the mutex/cv are touched only on the teardown handoff.
//
// Because we coordinate across threads that can call it at any arbitrary time,
// this object must have static lifetime.
class GlobalCallbackSession {
 public:
  GlobalCallbackSession() = default;
  ~GlobalCallbackSession() = default;

  // Single, static instance only: neither copyable nor movable.
  GlobalCallbackSession(const GlobalCallbackSession&) = delete;
  GlobalCallbackSession& operator=(const GlobalCallbackSession&) = delete;
  GlobalCallbackSession(GlobalCallbackSession&&) = delete;
  GlobalCallbackSession& operator=(GlobalCallbackSession&&) = delete;

  // Open the session (at enableProfiler time) so callbacks begin participating.
  // On a true enable a fresh RecordQueue was just installed, so bump the
  // session generation first: callbacks in the new session stamp the new value
  // and a straddling exit from the prior session detects the mismatch. The
  // mid-session dynamic collection toggle re-arms the same RecordQueue and
  // passes false; bumping there would wrongly drop end-events for ops that
  // straddle the toggle.
  void activate(bool new_session) {
    if (new_session) {
      generation_.fetch_add(1);
    }
    active_.store(true);
  }

  // Session generation, stamped into each callback's ObserverContext at enter
  // and re-checked at exit. A mismatch means the stamping session was torn down
  // (its RecordQueue and event freed), so the exit drops itself.
  uint64_t generation() const {
    return generation_.load();
  }

  // Whether the session is open; that is, teardown has not begun.
  bool isActive() const {
    return active_.load();
  }

  // Register an in-flight callback, paired with exit(). Callers must re-check
  // isActive() after enter() to close the race with a concurrent drain().
  void enter() {
    in_flight_.fetch_add(1);
  }

  // Deregister an in-flight callback. When this is the last one and the session
  // has been closed (drain() in progress), wake the waiting teardown thread.
  // During normal profiling this is a single lock-free decrement.
  void exit() {
    if (in_flight_.fetch_sub(1) == 1 && !active_.load()) {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_all();
    }
  }

  // Close the session to new callbacks and block until all in-flight ones have
  // exited; returns true once drained (also returns true if the session was
  // never opened; it remained purely thread-local profiling). Closing is fused
  // with the wait, so the session cannot be closed without waiting out
  // in-flight callbacks.
  //
  // Returns false if the drain does not complete within kDrainTimeout. A normal
  // drain takes microseconds (the in-flight window is just a callback's
  // getGlobal() + RecordQueue critical section, never an op body), so a
  // multi-second wait means a callback's thread was killed or wedged inside
  // that critical section. The caller must then leave the state installed:
  // freeing it under a live callback would reintroduce the use-after-free this
  // session guards against.
  bool drain() {
    if (!active_.exchange(false)) {
      return true;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(
        lock, kDrainTimeout, [this] { return in_flight_.load() == 0; });
  }

 private:
  // Backstop so disableProfiler() can never hang on a wedged or dead callback
  // thread. Far longer than any real drain, which completes in microseconds.
  static constexpr std::chrono::seconds kDrainTimeout{30};

  std::atomic<bool> active_{false};
  std::atomic<int64_t> in_flight_{0};
  std::atomic<uint64_t> generation_{0};
  std::mutex mutex_;
  std::condition_variable cv_; // guarded by mutex_
};

GlobalCallbackSession global_callback_session;

std::unique_ptr<at::ObserverContext> onFunctionEnterGlobal(
    const at::RecordFunction& fn) {
  // Fast bail once teardown has begun, before taking an in-flight ref, so the
  // drain in disableProfiler() always converges.
  if (!global_callback_session.isActive()) {
    return nullptr;
  }

  global_callback_session.enter();

  // Release the ref when this function returns (success or early bail), so the
  // in-flight window is just this enter's getGlobal() + begin_op() critical
  // section.
  auto in_flight_guard =
      c10::make_scope_exit([] { global_callback_session.exit(); });

  // Re-check after the increment. This is the handshake that makes teardown
  // safe: if disableProfiler() cleared active concurrently, its drain is
  // guaranteed to observe our increment and wait if and only if we still
  // observe active true here.
  if (!global_callback_session.isActive()) {
    return nullptr;
  }

  std::shared_ptr<KinetoThreadLocalState> state_ptr =
      KinetoThreadLocalState::getGlobal();
  if (!state_ptr) {
    return nullptr;
  }
  auto ctx = state_ptr->recordQueue.getSubqueue()->begin_op(fn);
  if (ctx) {
    // Stamp the current session so a straddling exit (begin_op here, exit after
    // this session is torn down) drops itself on mismatch instead of writing
    // through a freed event_.
    ctx->session_generation_ = global_callback_session.generation();
  }
  return ctx;
}

std::unique_ptr<at::ObserverContext> onFunctionEnterTLS(
    const at::RecordFunction& fn) {
  KinetoThreadLocalState* state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return nullptr;
  }
  return state_ptr->recordQueue.getSubqueue()->begin_op(fn);
}

void onFunctionExitImpl(
    KinetoThreadLocalState& state,
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  const auto& config = state.config();
  auto* kineto_ctx_ptr =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
  kineto_ctx_ptr->event_->end_time_ = c10::getApproximateTime();
  if (!config.experimental_config.performance_events.empty()) {
    state.recordQueue.getSubqueue()->disable_perf_profiler(
        *kineto_ctx_ptr->event_->counters_);
  }
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();
  if (fn.isNcclMeta()) {
    auto& extra_meta = *(kineto_ctx_ptr->event_->extra_nccl_meta_);
    // Record only the outputs in this exit callback of the record function
    torch::profiler::impl::SaveNcclMetaConfig ncclMetaConfig{
        true, false, false, true};
    auto additional_nccl_meta =
        torch::profiler::impl::saveNcclMeta(fn, ncclMetaConfig);
    extra_meta.insert(additional_nccl_meta.begin(), additional_nccl_meta.end());
  }
  if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      auto fallback = kineto_ctx_ptr->fallback_;
      TORCH_INTERNAL_ASSERT(fallback != nullptr);
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &fallback->device_event_end_, nullptr);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  } else if (config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    auto fallback = kineto_ctx_ptr->fallback_;
    TORCH_INTERNAL_ASSERT(fallback != nullptr);
    torch::profiler::impl::privateuse1Stubs()->record(
        nullptr, &fallback->device_event_end_, nullptr);
  }
}

// Pop the external correlation id that begin_op pushed for this op, if any.
// Must run on every onFunctionExit path, including the teardown and
// stale-session early exits that skip event finalization: the correlation stack
// lives in the device profiling backend (per thread) and is not reset across
// profiler sessions, so a skipped pop leaks. Safe on those paths because it
// touches only that stack, never the possibly-freed event_.
void maybePopCorrelationId(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  auto* kineto_ctx =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  if (kineto_ctx == nullptr || !kineto_ctx->pushed_correlation_id_) {
    return;
  }
  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::popUserCorrelationId();
  } else {
    torch::profiler::impl::kineto::popCorrelationId();
  }
}

void onFunctionExitGlobal(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  if (ctx_ptr == nullptr) {
    // Enter bailed (teardown in progress, or no state): no ref was taken and
    // there is nothing to finalize.
    return;
  }

  // Balance the correlation id pushed at begin_op on every path below,
  // including the teardown and stale-session early exits that skip event
  // finalization.
  auto correlation_guard =
      c10::make_scope_exit([&] { maybePopCorrelationId(fn, ctx_ptr); });

  // Take this exit's own in-flight ref, paired here because
  // onFunctionEnterGlobal no longer hands one off. It covers only this exit's
  // getGlobal() + RecordQueue write, never the op body, so the drain cannot
  // block on it.
  global_callback_session.enter();
  auto in_flight_guard =
      c10::make_scope_exit([] { global_callback_session.exit(); });

  // Re-check after the increment, mirroring onFunctionEnterGlobal: if teardown
  // began concurrently the queue may already be finalized/freed, so skip the
  // write (the event keeps its start and gets no end).
  if (!global_callback_session.isActive()) {
    return;
  }

  std::shared_ptr<KinetoThreadLocalState> state_ptr =
      KinetoThreadLocalState::getGlobal();
  if (!state_ptr) {
    return;
  }

  // If the session generations don't match, that means this op entered under an
  // earlier session that has since been torn down, freeing its RecordQueue and
  // event_. In that case, we don't need to do any exit cleanup.
  auto* kineto_ctx =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  if (kineto_ctx->session_generation_ != global_callback_session.generation()) {
    return;
  }

  onFunctionExitImpl(*state_ptr, fn, ctx_ptr);
}

void onFunctionExitTLS(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  // Balance the correlation id pushed at begin_op even when the TLS state is
  // gone by exit (early return below), for the same reason as
  // onFunctionExitGlobal.
  auto correlation_guard =
      c10::make_scope_exit([&] { maybePopCorrelationId(fn, ctx_ptr); });
  KinetoThreadLocalState* state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }
  onFunctionExitImpl(*state_ptr, fn, ctx_ptr);
}

void pushGlobalProfilingCallbacks(
    const std::unordered_set<at::RecordScope>& scopes,
    bool new_session) {
  std::shared_ptr<KinetoThreadLocalState> state_ptr =
      KinetoThreadLocalState::getGlobal();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto recordFunctionCallback =
      at::RecordFunctionCallback(onFunctionEnterGlobal, onFunctionExitGlobal)
          .needsInputs(state_ptr->config().report_input_shapes)
          .scopes(scopes);

  // Arm the drain gate before the global callback and fire on any thread. If
  // this a new profiling session, also bump the session generation.
  // disableProfiler() relies on this to know it must drain in-flight callbacks.
  global_callback_session.activate(new_session);

  state_ptr->setCallbackHandle(at::addGlobalCallback(recordFunctionCallback));
}

void pushTLSProfilingCallbacks(
    const std::unordered_set<at::RecordScope>& scopes) {
  KinetoThreadLocalState* state_ptr = KinetoThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto recordFunctionCallback =
      at::RecordFunctionCallback(onFunctionEnterTLS, onFunctionExitTLS)
          .needsInputs(state_ptr->config().report_input_shapes)
          .scopes(scopes);
  state_ptr->setCallbackHandle(
      at::addThreadLocalCallback(recordFunctionCallback));
}

struct ProfilerStateInfo {
  std::shared_ptr<KinetoThreadLocalState> state_ptr;
  std::unordered_set<at::RecordScope> scopes;
};
std::shared_ptr<ProfilerStateInfo> profiler_state_info_ptr{nullptr};

} // namespace

void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  TORCH_INTERNAL_ASSERT(
      KinetoThreadLocalState::getGlobal() == nullptr,
      "On-demand profiling does not support post processing callback");

  KinetoThreadLocalState* state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }

  state_ptr->recordQueue.getSubqueue()->emplace_backend_event(
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
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const ActivityFilter& activity_filter) {
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::ITT) {
    return;
  }

  // Forward registered PrivateUse1 profiler factory to Kineto.
  // Only for KINETO_PRIVATEUSE1 state where backend provides its own
  // IActivityProfiler.
#ifdef USE_KINETO
  if (config.state == ProfilerState::KINETO_PRIVATEUSE1) {
    torch::profiler::impl::PrivateUse1ProfilerRegistry::instance()
        .onKinetoInit();
  }
#endif // USE_KINETO

  TORCH_CHECK(
      isKinetoCompatibleState(config.state),
      "Supported only in Kineto profiler");
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!hasRequestedDeviceActivity(activities),
      activities,
      config.experimental_config,
      config.trace_id,
      activity_filter);

  if (!config.experimental_config.performance_events.empty()) {
    /* For now only CPU activity is supported */
    TORCH_CHECK(
        activities.count(torch::autograd::profiler::ActivityType::CPU),
        "Cannot run cpu hardware profiler without CPU activities, please only use CPU activity type");
    /*
     * Sending a warning and passing the non-standard event to the backend
     * Backend can abort if the event is not supported.
     * TODO Should we gracefully drop the invalid event if we have at least one
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

static void toggleTorchOpCollectionDynamic(bool enable) {
  std::shared_ptr<ProfilerStateBase> global_state =
      ProfilerStateBase::getGlobal();
  if (global_state) {
    if (enable) {
      auto scopes = profiler_state_info_ptr->scopes;
      // Mid-session re-arm on the same RecordQueue: do not bump the generation.
      pushGlobalProfilingCallbacks(scopes, /*new_session=*/false);
    } else {
      global_state->removeCallback();
    }
  } else {
    ProfilerStateBase* tls_state = ProfilerStateBase::getTLS();
    TORCH_CHECK(tls_state);
    if (enable) {
      auto scopes = profiler_state_info_ptr->scopes;
      pushTLSProfilingCallbacks(scopes);
    } else {
      tls_state->removeCallback();
    }
  }
}

// Set this function to be unused as profiler implementation needs more
// refactoring to support Python ops collection dynamic toggling
#ifdef _MSC_VER
#define UNUSED
#else
#define UNUSED __attribute__((unused))
#endif
static UNUSED void togglePythonCollectionDynamic(bool enable) {
  std::shared_ptr<KinetoThreadLocalState> global_state =
      KinetoThreadLocalState::getGlobal();
  if (global_state) {
    if (enable) {
      global_state->resumePython();
    } else {
      global_state->pausePython();
    }
  } else {
    KinetoThreadLocalState* tls_state = KinetoThreadLocalState::getTLS();
    TORCH_CHECK(tls_state);
    if (enable) {
      tls_state->resumePython();
    } else {
      tls_state->pausePython();
    }
  }
}

static void toggleCPUCollectionDynamic(bool enable) {
  toggleTorchOpCollectionDynamic(enable);
  // For now we only support Torch Op collection dynamic toggling as
  // implementing Python ops would require not only string parsing to get rid of
  // the toggling events as well as other unfinished events as well as changes
  // in stack logic
  // togglePythonCollectionDynamic(enable);
}

void toggleCollectionDynamic(
    const bool enable,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (activities.contains(torch::autograd::profiler::ActivityType::CPU) &&
      (!activities.contains(torch::autograd::profiler::ActivityType::CUDA) ||
       !activities.contains(torch::autograd::profiler::ActivityType::XPU))) {
    LOG(WARNING)
        << "Toggling CPU activity with GPU activity on may result in traces with GPU events on arbitrary tracks";
  } else if (
      (activities.contains(torch::autograd::profiler::ActivityType::CUDA) ||
       activities.contains(torch::autograd::profiler::ActivityType::XPU)) &&
      !activities.contains(torch::autograd::profiler::ActivityType::CPU)) {
    LOG(WARNING)
        << "Toggling GPU activity with CPU activity on may result in traces with incorrect correlation between CPU and GPU events";
  }
  for (auto act : activities) {
    if (act == torch::autograd::profiler::ActivityType::CUDA ||
        act == torch::autograd::profiler::ActivityType::XPU) {
      torch::profiler::impl::kineto::toggleCollectionDynamic(enable);
    } else if (act == torch::autograd::profiler::ActivityType::CPU) {
      toggleCPUCollectionDynamic(enable);
    } else {
      LOG(WARNING)
          << "Dynamic toggle is only supported for CPU/GPU activity, skipping toggling of "
          << actToString(act);
      continue;
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
      KinetoThreadLocalState::getGlobal() == nullptr,
      "On-demand profiling does not support post processing callback");

  enableProfiler(config, activities, scopes);
  if (config.pushGlobalCallbacks()) {
    std::shared_ptr<KinetoThreadLocalState> state_ptr =
        KinetoThreadLocalState::getGlobal();
    state_ptr->setEventPostProcessingCallback(std::move(cb));
  } else {
    KinetoThreadLocalState* state_ptr = KinetoThreadLocalState::getTLS();
    state_ptr->setEventPostProcessingCallback(std::move(cb));
  }
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  const auto has_cpu = activities.count(ActivityType::CPU);
  bool already_enabled = config.pushGlobalCallbacks()
      ? KinetoThreadLocalState::getGlobal() != nullptr
      : KinetoThreadLocalState::getTLS() != nullptr;
  TORCH_CHECK(
      !already_enabled,
      "Profiler is already enabled",
      (config.global() ? "." : " on this thread."));

  // Handle external tracer states - these use their own tracing callbacks
  if (isExternalTracerState(config.state)) {
    switch (config.state) {
      case ProfilerState::NVTX:
        torch::profiler::impl::pushNVTXCallbacks(config, scopes);
        break;
      case ProfilerState::ITT:
        torch::profiler::impl::pushITTCallbacks(config, scopes);
        break;
      case ProfilerState::PRIVATEUSE1:
        torch::profiler::impl::pushPRIVATEUSE1CallbacksStub(config, scopes);
        break;
      default:
        break;
    }
    return;
  }

  TORCH_CHECK(isKinetoCompatibleState(config.state) || config.global());
  TORCH_CHECK(!activities.empty(), "No activities specified.");
  TORCH_INTERNAL_ASSERT(
      has_cpu || !config.global(),
      "Ondemand profiling must enable CPU tracing");

  auto state_ptr = std::make_shared<KinetoThreadLocalState>(config, activities);
  KinetoThreadLocalState::push(state_ptr);

  if (has_cpu) {
    config.pushGlobalCallbacks()
        ? pushGlobalProfilingCallbacks(scopes, /*new_session=*/true)
        : pushTLSProfilingCallbacks(scopes);
  }

  if (!config.global()) {
    torch::profiler::impl::kineto::startTrace();
  }

  if (has_cpu) {
    auto state_info_ptr = std::make_shared<ProfilerStateInfo>();
    state_info_ptr->state_ptr = std::move(state_ptr);
    state_info_ptr->scopes = scopes;
    profiler_state_info_ptr = std::move(state_info_ptr);
  }
}

bool isProfilerEnabledInMainThread() {
  return profiler_state_info_ptr != nullptr;
}

void enableProfilerInChildThread() {
  auto state_info_ptr = profiler_state_info_ptr;
  TORCH_CHECK(state_info_ptr, "Profiler is not enabled in main thread.");
  TORCH_CHECK(
      KinetoThreadLocalState::getTLS() == nullptr,
      "Profiler is already enabled in this thread.");

  KinetoThreadLocalState::push(state_info_ptr->state_ptr);
  pushTLSProfilingCallbacks(state_info_ptr->scopes);
}

void disableProfilerInChildThread() {
  auto state_ptr = ProfilerStateBase::pop();
  TORCH_CHECK(
      state_ptr,
      "Can't disable Kineto profiler when it's not running in this thread");
  state_ptr->removeCallback();
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  // releasing to inform child threads to stop profiling
  profiler_state_info_ptr = nullptr;

  // If global callbacks were installed (KINETO_ONDEMAND, or KINETO with
  // profile_all_threads), they may be running on other threads right now. Wait
  // for in-flight ones to finish before popping and finalizing the state, else
  // a worker thread can mutate the record queue while finalizeTrace() reads
  // and frees it.
  if (!global_callback_session.drain()) {
    // The drain timed out: an in-flight global callback is wedged or dead, so
    // we cannot finalize (reading the queue would race the live callback and
    // reintroduce the use-after-free). We still pop the state and remove the
    // global callback so a later enableProfiler() is not rejected as "already
    // enabled". Safe with callbacks in flight: the shared_ptr keepalive keeps
    // the state alive for one still using it, and removeCallback() is versioned
    // (running invocations use per-thread snapshots). The session trace is
    // lost.
    LOG(ERROR)
        << "disableProfiler timed out draining in-flight global RecordFunction "
           "callbacks; abandoning this trace. A callback thread is likely "
           "wedged or was killed mid-callback.";
    if (auto state_ptr = ProfilerStateBase::pop()) {
      state_ptr->removeCallback();
    }
    return std::make_unique<ProfilerResult>();
  }

  auto state_ptr = ProfilerStateBase::pop();
  if (!state_ptr) {
    LOG(WARNING)
        << "disableProfiler called but no active profiling session found. "
        << "This can happen if profiling was cancelled during warmup.";
    return std::make_unique<ProfilerResult>();
  }
  const auto& config = state_ptr->config();
  TORCH_CHECK(
      isValidDisableState(config.state),
      "Can't disable Kineto profiler: config is not in a valid disable state");

  state_ptr->removeCallback();

  // Traces are converged via libkineto automatically for ondemand flow
  if (config.global()) {
    (void)std::static_pointer_cast<KinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  // Shared among NVTX, PRIVATEUSE1, KINETO, KINETO_GPU_FALLBACK,
  // KINETO_PRIVATEUSE1_FALLBACK
  std::unique_ptr<ProfilerResult> result;
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::PRIVATEUSE1) {
    result = std::make_unique<ProfilerResult>();
  }

  if (isKinetoCompatibleState(config.state)) {
    auto kineto_state_ptr =
        std::static_pointer_cast<KinetoThreadLocalState>(state_ptr);
    auto trace = kineto_state_ptr->finalizeTrace();
    result = std::make_unique<ProfilerResult>(
        kineto_state_ptr->startTime,
        std::move(kineto_state_ptr->kinetoEvents),
        std::move(trace),
        std::move(kineto_state_ptr->eventTree));
  }

  return result;
}
namespace tracer = torch::profiler::impl::python_tracer;
static std::unique_ptr<tracer::PythonMemoryTracerBase> memory_tracer;
void startMemoryProfile() {
  if (memory_tracer == nullptr) {
    memory_tracer = tracer::PythonMemoryTracerBase::make();
  }
  memory_tracer->start();
}

void stopMemoryProfile() {
  memory_tracer->stop();
}

void exportMemoryProfile(const std::string& filename) {
  memory_tracer->export_memory_history(filename);
}

KinetoEvent::KinetoEvent(
    const std::shared_ptr<const torch::profiler::impl::Result>& result,
    const bool verbose)
    : result_{result} {
  TORCH_INTERNAL_ASSERT(result != nullptr);

  if (verbose) {
    // Populate Python stack
    auto parent = result_->parent_.lock();
    while (parent != nullptr) {
      parent->visit_if_base<PyExtraFieldsBase>(
          [&](const auto&) { python_stack_.push_back(parent->name()); });
      parent = parent->parent_.lock();
    }
  }

  result->visit_if_base<ExtraFields<EventType::TorchOp>>([&](const auto& op) {
    auto arg_data = parseArgData(op.inputs_, op.concrete_inputs_);
    shapes_ = std::move(arg_data.shapesForKinetoEvent);
    structured_input_shapes_ = std::move(arg_data.shapes);
    structured_input_strides_ = std::move(arg_data.strides);
    dtypes_ = std::move(arg_data.dtypes);
    concrete_inputs_ = std::move(arg_data.concreteInputs);
    kwinputs_ = std::move(op.kwinputs_);
  });
}

bool KinetoEvent::isPythonFunction() const {
  bool out{false};
  result_->visit_if_base<PyExtraFieldsBase>([&](const auto&) { out = true; });
  return out;
}

int64_t KinetoEvent::pythonId() const {
  int64_t out{-1};
  result_->visit_if_base<PyExtraFieldsBase>(
      [&](const auto& i) { out = static_cast<int64_t>(i.id_); });
  return out;
}

int64_t KinetoEvent::pythonParentId() const {
  int64_t out{-1};
  // Walk the python parent pointers up to find the next event of type
  // PyExtraFieldsBase
  result_->visit_if_base<PyExtraFieldsBase>([&](const auto&) {
    auto parent = result_->parent_.lock();
    while (parent) {
      parent->visit_if_base<PyExtraFieldsBase>(
          [&](const auto& j) { out = static_cast<int64_t>(j.id_); });
      if (out >= 0) {
        break;
      }
      parent = parent->parent_.lock();
    }
  });
  return out;
}

int64_t KinetoEvent::pythonModuleId() const {
  int64_t out{-1};
  // Returns the module id for PyCall events (python function calls to
  // nn.Module)
  result_->visit(c10::overloaded(
      [&](const ExtraFields<EventType::PyCall>& py_call) {
        if (py_call.module_.has_value()) {
          out = static_cast<int64_t>(py_call.module_->id_);
        }
      },
      [](const auto&) {}));
  return out;
}

bool KinetoEvent::hasShapes() const {
  return !shapes_.empty();
}

const c10::ArrayRef<std::vector<int64_t>> KinetoEvent::shapes() const {
  return shapes_;
}

const c10::ArrayRef<torch::profiler::impl::shape> KinetoEvent::
    structuredInputShapes() const {
  return structured_input_shapes_;
}

const c10::ArrayRef<torch::profiler::impl::shape> KinetoEvent::
    structuredInputStrides() const {
  return structured_input_strides_;
}

bool KinetoEvent::hasTypes() const {
  return !dtypes_.empty();
}

const c10::ArrayRef<std::string> KinetoEvent::dtypes() const {
  return dtypes_;
}

bool KinetoEvent::hasConcreteInputs() const {
  return !concrete_inputs_.empty();
}

const c10::ArrayRef<c10::IValue> KinetoEvent::concreteInputs() const {
  return concrete_inputs_;
}

bool KinetoEvent::hasKwinputs() const {
  return !kwinputs_.empty();
}

bool KinetoEvent::isHiddenEvent() const {
  return result_ && result_->hidden_;
}

const std::unordered_map<std::string, c10::IValue> KinetoEvent::kwinputs()
    const {
  return kwinputs_;
}

const c10::ArrayRef<std::string> KinetoEvent::stack() const {
  auto get = [&](const auto& i) -> auto& {
    return !i.jit_stack_.empty() ? i.jit_stack_ : python_stack_;
  };

  auto const& extra_fields = result_->extra_fields_;
  if (auto p = std::get_if<ExtraFields<EventType::TorchOp>>(&extra_fields)) {
    return get(*p);
  }
  if (auto p = std::get_if<ExtraFields<EventType::Backend>>(&extra_fields)) {
    return get(*p);
  }
  return python_stack_;
}

const c10::ArrayRef<std::string> KinetoEvent::moduleHierarchy() const {
  auto const& extra_fields = result_->extra_fields_;
  if (auto p = std::get_if<ExtraFields<EventType::TorchOp>>(&extra_fields)) {
    return p->jit_modules_;
  }
  if (auto p = std::get_if<ExtraFields<EventType::Backend>>(&extra_fields)) {
    return p->jit_modules_;
  }
  return {};
}

uint64_t KinetoEvent::endNs() const {
  return result_->endTimeNS();
}

uint64_t KinetoEvent::durationNs() const {
  return (result_->endTimeNS() - result_->start_time_ns_);
}

int64_t KinetoEvent::debugHandle() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& i) { return i.debug_handle_; },
      [](const ExtraFields<EventType::Backend>& i) { return i.debug_handle_; },
      [](const auto&) -> int64_t { return -1; }));
}

int KinetoEvent::deviceIndex() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::Allocation>& i) {
        return static_cast<int>(i.device_index_);
      },
      [](const ExtraFields<EventType::OutOfMemory>& i) {
        return static_cast<int>(i.device_index_);
      },
      [&](const auto&) {
        return static_cast<int>(result_->kineto_info_.device);
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

int64_t KinetoEvent::privateuse1ElapsedUs() const {
  auto privateuse1_event_start = fallbackStart();
  auto privateuse1_event_end = fallbackEnd();
  if (!privateuse1_event_start || !privateuse1_event_end) {
    return -1;
  }
  return (int64_t)torch::profiler::impl::privateuse1Stubs()->elapsed(
      &privateuse1_event_start, &privateuse1_event_end);
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

std::string KinetoEvent::metadataJson() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& op) -> std::string {
        return op.metadata_json_;
      },
      [](const ExtraFields<EventType::Kineto>& op) -> std::string {
        return op.metadata_json_;
      },
      [](const auto&) -> std::string { return std::string(""); }));
}

int64_t KinetoEvent::externalId() const {
  // Mirrors libkineto::ChromeTraceLogger::handleActivity() "External id" logic.
  // libkineto::ChromeTraceLogger checks op.linkedActivity() != nullptr; here we
  // check linkedCorrelationId() > 0, which is equivalent because PyTorch
  // correlation IDs are monotonically increasing from 1 (a valid linked
  // activity always has a non-zero correlation ID).
  uint64_t linked = linkedCorrelationId();
  if (linked > 0) {
    return static_cast<int64_t>(linked);
  }

  // Orphaned GPU activities (no linked CPU op) in these types should not get
  // an External id, to avoid incorrect cross-linking in trace viewers.
  auto type = static_cast<libkineto::ActivityType>(activityType());
  if (type != libkineto::ActivityType::GPU_MEMCPY &&
      type != libkineto::ActivityType::GPU_MEMSET &&
      type != libkineto::ActivityType::CONCURRENT_KERNEL &&
      type != libkineto::ActivityType::CUDA_RUNTIME &&
      type != libkineto::ActivityType::CUDA_DRIVER &&
      type != libkineto::ActivityType::PRIVATEUSE1_RUNTIME &&
      type != libkineto::ActivityType::PRIVATEUSE1_DRIVER) {
    return static_cast<int64_t>(result_->visit(c10::overloaded(
        [](const ExtraFields<EventType::TorchOp>& e) -> uint64_t {
          return e.correlation_id_;
        },
        [](const ExtraFields<EventType::Kineto>& e) -> uint64_t {
          return e.correlation_id_;
        },
        [](const auto&) -> uint64_t { return 0; })));
  }

  return 0;
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
FORWARD_FROM_RESULT(overload_name, overload_name())
FORWARD_FROM_RESULT(deviceType, deviceType())
FORWARD_FROM_RESULT(startNs, start_time_ns_)
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
TYPED_ATTR(TorchOp, scope, static_cast<uint8_t>(e.scope_))
TYPED_ATTR(TorchOp, hasModuleHierarchy, !e.jit_modules_.empty())
TYPED_ATTR(TorchOp, isAsync, e.is_async_)

extra_meta_t KinetoEvent::extraMeta() const {
  extra_meta_t out;
  result_->visit(c10::overloaded(
      [&](const ExtraFields<EventType::TorchOp>& e) { out = e.extra_meta_; },
      [&](const ExtraFields<EventType::Kineto>& e) { out = e.extra_meta_; },
      [](const auto&) {}));
  return out;
}

TYPED_ATTR(TorchOp, fallbackStart, e.device_fallback_.device_event_start_)
TYPED_ATTR(TorchOp, fallbackEnd, e.device_fallback_.device_event_end_)
TYPED_ATTR(
    TorchOp,
    flops,
    !e.extra_args_.empty()
        ? torch::profiler::impl::computeFlops(e.name_, e.extra_args_)
        : 0)
TYPED_ATTR(Backend, backend, e.backend_)
TYPED_ATTR(Allocation, nBytes, e.alloc_size_)
TYPED_ATTR(Kineto, linkedCorrelationId, [&]() {
  const auto linked = e.linked_activity_.lock();
  return linked ? linked->correlationID() : 0;
}())
#undef TYPED_ATTR
#undef TYPED_ATTR_WITH_DEFAULT

// Flow fields exist on both TorchOp and Kineto event types.
uint32_t KinetoEvent::flowId() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& e) { return e.flow.id; },
      [](const ExtraFields<EventType::Kineto>& e) { return e.flow.id; },
      [](const auto&) -> uint32_t { return 0; }));
}
uint32_t KinetoEvent::flowType() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& e) { return e.flow.type; },
      [](const ExtraFields<EventType::Kineto>& e) { return e.flow.type; },
      [](const auto&) -> uint32_t { return 0; }));
}
bool KinetoEvent::flowStart() const {
  return result_->visit(c10::overloaded(
      [](const ExtraFields<EventType::TorchOp>& e) {
        return static_cast<bool>(e.flow.start);
      },
      [](const ExtraFields<EventType::Kineto>& e) {
        return static_cast<bool>(e.flow.start);
      },
      [](const auto&) { return false; }));
}

ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&&
        trace,
    std::vector<experimental_event_t>&& event_tree)
    : trace_start_ns_(start_time),
      events_(std::move(events)),
      trace_(std::move(trace)),
      event_tree_(std::move(event_tree)) {}
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  trace_->save(path);
}

#ifdef USE_KINETO
const std::vector<const libkineto::ITraceActivity*>* ProfilerResult::
    traceActivities() {
  if (trace_) {
    return trace_->get()->activities();
  }
  return nullptr;
}
#endif

} // namespace autograd::profiler

namespace profiler::impl {
void _reportVulkanEventToProfiler(vulkan_id_t id) {
  ::torch::autograd::profiler::KinetoThreadLocalState* state_ptr =
      ::torch::autograd::profiler::KinetoThreadLocalState::getTLS();
  if (state_ptr) {
    state_ptr->reportVulkanEventToProfiler(id);
  }
}
} // namespace profiler::impl

} // namespace torch
