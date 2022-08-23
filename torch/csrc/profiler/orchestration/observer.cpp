#include <torch/csrc/profiler/orchestration/observer.h>

#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

ExperimentalConfig::ExperimentalConfig(
    std::vector<std::string> profiler_metrics,
    bool profiler_measure_per_kernel)
    : profiler_metrics{profiler_metrics},
      profiler_measure_per_kernel{profiler_measure_per_kernel} {}

ExperimentalConfig::operator bool() const {
  return !profiler_metrics.empty();
}

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

/*explicit*/ ProfilerThreadLocalStateBase::ProfilerThreadLocalStateBase(
    const ProfilerConfig& config)
    : c10::MemoryReportingInfoBase(), config_(config) {}

ProfilerThreadLocalStateBase::~ProfilerThreadLocalStateBase() {
  if (handle_) {
    auto handle = handle_;
    removeCallback();
    SOFT_ASSERT(false, "Leaked callback handle: ", handle);
  }
}

void ProfilerThreadLocalStateBase::setCallbackHandle(
    at::CallbackHandle handle) {
  if (handle_) {
    at::removeCallback(handle_);
    SOFT_ASSERT(
        false,
        "ProfilerStateBase already has a registered callback. "
        "Removing to avoid leaked callback.");
  }

  handle_ = handle;
}

void ProfilerThreadLocalStateBase::removeCallback() {
  if (handle_) {
    at::removeCallback(handle_);
    handle_ = 0;
  }
}

bool profilerEnabled() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  return state_ptr && !state_ptr->config().disabled();
}

TORCH_API ActiveProfilerType profilerType() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}

torch::profiler::impl::ProfilerConfig getProfilerConfig() {
  auto state_ptr = ProfilerThreadLocalStateBase::getTLS();
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

} // namespace impl
} // namespace profiler
} // namespace torch
