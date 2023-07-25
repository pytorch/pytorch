#include <torch/csrc/profiler/orchestration/observer.h>

#include <torch/csrc/profiler/util.h>

#include <utility>

namespace torch {
namespace profiler {
namespace impl {

using GlobalManager = GlobalStateManager<ProfilerStateBase>;

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
ExperimentalConfig::ExperimentalConfig(
    std::vector<std::string> profiler_metrics,
    bool profiler_measure_per_kernel,
    bool verbose,
    std::vector<std::string> performance_events,
    bool adjust_timestamps)
    : profiler_metrics{std::move(profiler_metrics)},
      profiler_measure_per_kernel{profiler_measure_per_kernel},
      verbose{verbose},
      performance_events(std::move(performance_events)),
      adjust_timestamps{adjust_timestamps} {}

/*explicit*/ ExperimentalConfig::operator bool() const {
  return !profiler_metrics.empty();
}

ProfilerConfig::ProfilerConfig(
    ProfilerState state,
    bool report_input_shapes,
    bool profile_memory,
    bool with_stack,
    bool with_flops,
    bool with_modules,
    ExperimentalConfig experimental_config)
    : state{state},
      experimental_config{std::move(experimental_config)},
      report_input_shapes{report_input_shapes},
      profile_memory{profile_memory},
      with_stack{with_stack},
      with_flops{with_flops},
      with_modules{with_modules} {}

bool ProfilerConfig::disabled() const {
  return state == torch::profiler::impl::ProfilerState::Disabled;
}

bool ProfilerConfig::global() const {
  return state == torch::profiler::impl::ProfilerState::KINETO_ONDEMAND;
}

namespace {
enum ProfilerIValueIdx {
  STATE = 0,
  REPORT_INPUT_SHAPES,
  PROFILE_MEMORY,
  NUM_PROFILER_CFG_IVALUE_IDX // must be last in list
};
} // namespace

at::IValue ProfilerConfig::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(state));
  eventIValueList.emplace_back(report_input_shapes);
  eventIValueList.emplace_back(profile_memory);
  return eventIValueList;
}

ProfilerConfig ProfilerConfig::fromIValue(
    const at::IValue& profilerConfigIValue) {
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = profilerConfigIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
}

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
/*explicit*/ ProfilerStateBase::ProfilerStateBase(const ProfilerConfig& config)
    : c10::MemoryReportingInfoBase(), config_(config) {}

ProfilerStateBase::~ProfilerStateBase() {
  if (handle_) {
    auto handle = handle_;
    removeCallback();
    SOFT_ASSERT(false, "Leaked callback handle: ", handle);
  }
}

/*static*/ ProfilerStateBase* ProfilerStateBase::get(bool global) {
  auto* out = global
      ? GlobalManager::get()
      : static_cast<ProfilerStateBase*>(
            c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

/*static*/ void ProfilerStateBase::push(
    std::shared_ptr<ProfilerStateBase>&& state) {
  TORCH_INTERNAL_ASSERT(state != nullptr);
  if (state->config().global()) {
    GlobalManager::push(std::move(state));
  } else {
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  }
}

namespace {
std::shared_ptr<ProfilerStateBase> popTLS() {
  // If there is no active thread local profiler then we simply return null.
  // However if there is an active profiler but it is not the top
  // `DebugInfoBase`then `c10::ThreadLocalDebugInfo::_pop` will throw.
  // TODO(robieta): make `noexcept` version.
  return c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
      ? std::static_pointer_cast<ProfilerStateBase>(
            c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE))
      : nullptr;
}
} // namespace

/*static*/ std::shared_ptr<ProfilerStateBase> ProfilerStateBase::pop(
    bool global) {
  auto out = global ? GlobalManager::pop() : popTLS();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

void ProfilerStateBase::setCallbackHandle(at::CallbackHandle handle) {
  if (handle_) {
    at::removeCallback(handle_);
    SOFT_ASSERT(
        false,
        "ProfilerStateBase already has a registered callback. "
        "Removing to avoid leaked callback.");
  }

  handle_ = handle;
}

void ProfilerStateBase::removeCallback() {
  if (handle_) {
    at::removeCallback(handle_);
    handle_ = 0;
  }
}

bool profilerEnabled() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr && !state_ptr->config().disabled();
}

TORCH_API ActiveProfilerType profilerType() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}

torch::profiler::impl::ProfilerConfig getProfilerConfig() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

} // namespace impl
} // namespace profiler
} // namespace torch
