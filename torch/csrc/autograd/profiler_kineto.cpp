#include <cstring>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>

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

#include <ATen/Context.h>

#include <stdexcept>
#include <utility>

#ifdef USE_KINETO
#include <ApproximateClock.h>
#include <libkineto.h>
#include <time_since_epoch.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type();
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
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
    return OpArgData{false, {}, {}, {}, {}, {}};
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
      true,
      shapes,
      dtypes,
      concrete_inputs_list,
      shapesForKinetoEvent,
      strides};
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

  void addMetadata(const std::string& key, const std::string& value) {
    if (kinetoActivity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<torch::profiler::impl::kineto::activity_t*>(
              kinetoActivity_),
          key,
          value);
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
    const auto arg_data =
        parseArgData(op_event.inputs_, op_event.concrete_inputs_);

    if (arg_data.hasData) {
      if (get_record_concrete_inputs_enabled()) {
        addMetadata("Input Dims", variantShapesToStr(arg_data.shapes));
        addMetadata("Input Strides", variantShapesToStr(arg_data.strides));
      } else {
        addMetadata("Input Dims", shapesToStr(arg_data.shapesForKinetoEvent));
      }
      addMetadata("Input type", strListToStr(arg_data.dtypes));
      if (!arg_data.concreteInputs.empty()) {
        addMetadata(
            "Concrete Inputs", ivalueListToStr(arg_data.concreteInputs));
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
  void operator()(const T&) {}

 private:
  /* To get names of the performance events */
  const torch::profiler::impl::ProfilerConfig* config_;
};

struct KinetoThreadLocalState : public ProfilerStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerStateBase(config),
        startTime(getTimeNs()),
        recordQueue(config, std::move(activities)) {}
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

    materializeOpEvents(records_and_trace.first);

    // `kinetoEvents` does not include Python events. Instead it exposes them
    // via the `stacks` property.
    kinetoEvents.erase(
        std::remove_if(
            kinetoEvents.begin(),
            kinetoEvents.end(),
            [](const auto& i) { return i.isPythonFunction(); }),
        kinetoEvents.end());

    return std::move(records_and_trace.second);
  }

  template <typename T>
  void invokeCallback(T& t) {
    if (eventPostProcessCb) {
      eventPostProcessCb(t.debug_handle_, t.jit_stack_, t.jit_modules_);
    }
  }

  void materializeOpEvents(std::vector<std::shared_ptr<Result>>& events) {
    for (auto& e : events) {
      if (e->parent_.expired() && e->deviceType() == c10::DeviceType::CPU) {
        eventTree.push_back(e);
      }

      if (e->finished_) {
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
  }

  uint64_t startTime;
  c10::ApproximateClockToUnixTimeConverter clockConverter;
  torch::profiler::impl::RecordQueue recordQueue;
  std::vector<KinetoEvent> kinetoEvents;
  std::vector<experimental_event_t> eventTree;
  // Optional, if event post-processing is enabled.
  post_process_t eventPostProcessCb;
};

template <bool use_global_state_ptr = false>
std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  auto state_ptr = KinetoThreadLocalState::get(use_global_state_ptr);
  if (!state_ptr) {
    return nullptr;
  }
  return state_ptr->recordQueue.getSubqueue()->begin_op(fn);
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
  kineto_ctx_ptr->event_->end_time_ = c10::getApproximateTime();
  if (!config.experimental_config.performance_events.empty()) {
    state_ptr->recordQueue.getSubqueue()->disable_perf_profiler(
        *kineto_ctx_ptr->event_->counters_);
  }
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();
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

  if constexpr (use_global_callback) {
    registration_state_ptr->setCallbackHandle(
        at::addGlobalCallback(recordFunctionCallback));
  } else {
    registration_state_ptr->setCallbackHandle(
        at::addThreadLocalCallback(recordFunctionCallback));
  }
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
      KinetoThreadLocalState::get(/*global=*/true) == nullptr,
      "On-demand profiling does not support post processing callback");

  auto state_ptr = KinetoThreadLocalState::get(/*global=*/false);
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
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::ITT) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK ||
          config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK,
      "Supported only in Kineto profiler");
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!(
          at::hasCUDA() || at::hasXPU() || at::hasMTIA() ||
          c10::get_privateuse1_backend() != "privateuseone"),
      activities,
      config.experimental_config);

  if (!config.experimental_config.performance_events.empty()) {
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
  } else if (config.state == ProfilerState::PRIVATEUSE1) {
    torch::profiler::impl::pushPRIVATEUSE1CallbacksStub(config, scopes);
    return;
  }

  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK ||
      config.global());
  TORCH_CHECK(!activities.empty(), "No activities specified.");
  TORCH_INTERNAL_ASSERT(
      has_cpu || !config.global(),
      "Ondemand profiling must enable CPU tracing");

  auto state_ptr = std::make_shared<KinetoThreadLocalState>(config, activities);
  KinetoThreadLocalState::push(state_ptr);

  if (has_cpu) {
    config.global() ? pushProfilingCallbacks</*global=*/true>(scopes)
                    : pushProfilingCallbacks</*global=*/false>(scopes);
  }

  if (!config.global()) {
    torch::profiler::impl::kineto::startTrace();
  }

  if (has_cpu) {
    auto state_info_ptr = std::make_shared<ProfilerStateInfo>();
    state_info_ptr->state_ptr = state_ptr;
    state_info_ptr->scopes = scopes;
    profiler_state_info_ptr = state_info_ptr;
  }
}

bool isProfilerEnabledInMainThread() {
  return profiler_state_info_ptr != nullptr;
}

void enableProfilerInChildThread() {
  auto state_info_ptr = profiler_state_info_ptr;
  TORCH_CHECK(state_info_ptr, "Profiler is not enabled in main thread.");
  TORCH_CHECK(
      KinetoThreadLocalState::get(/*global=*/false) == nullptr,
      "Profiler is already enabled in this thread.");

  KinetoThreadLocalState::push(state_info_ptr->state_ptr);
  pushProfilingCallbacks</*global=*/false>(state_info_ptr->scopes);
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

  auto state_ptr = ProfilerStateBase::pop();
  const auto& config = state_ptr->config();
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK ||
           config.state == ProfilerState::KINETO_ONDEMAND ||
           config.state == ProfilerState::NVTX ||
           config.state == ProfilerState::ITT ||
           config.state == ProfilerState::PRIVATEUSE1),
      "Can't disable Kineto profiler when it's not running");

  state_ptr->removeCallback();

  // Traces are converged via libkineto automatically for ondemand flow
  if (state_ptr->config().global()) {
    (void)std::static_pointer_cast<KinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  // Shared among NVTX, PRIVATEUSE1, KINETO, KINETO_GPU_FALLBACK,
  // KINETO_PRIVATEUSE1_FALLBACK
  std::unique_ptr<ProfilerResult> result;
  if (state_ptr->config().state == ProfilerState::NVTX ||
      state_ptr->config().state == ProfilerState::PRIVATEUSE1) {
    result = std::make_unique<ProfilerResult>();
  }

  if (config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
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
    dtypes_ = std::move(arg_data.dtypes);
    concrete_inputs_ = std::move(arg_data.concreteInputs);
  });
}

bool KinetoEvent::isPythonFunction() const {
  bool out{false};
  result_->visit_if_base<PyExtraFieldsBase>([&](const auto&) { out = true; });
  return out;
}

bool KinetoEvent::hasShapes() const {
  return !shapes_.empty();
}

const c10::ArrayRef<std::vector<int64_t>> KinetoEvent::shapes() const {
  return shapes_;
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
TYPED_ATTR(TorchOp, extraMeta, e.extra_meta_)
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

} // namespace autograd::profiler

namespace profiler::impl {
void _reportVulkanEventToProfiler(vulkan_id_t id) {
  auto state_ptr = ::torch::autograd::profiler::KinetoThreadLocalState::get(
      /*global=*/false);
  if (state_ptr) {
    state_ptr->reportVulkanEventToProfiler(id);
  }
}
} // namespace profiler::impl

} // namespace torch
