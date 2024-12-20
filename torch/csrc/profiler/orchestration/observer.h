#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/Export.h>

#include <utility>

namespace torch::profiler::impl {

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
enum class C10_API_ENUM ActivityType {
  CPU = 0,
  XPU, // XPU kernels, runtime
  CUDA, // CUDA kernels, runtime
  MTIA, // MTIA kernels, runtime
  PrivateUse1, // PrivateUse1 kernels, runtime
  NUM_KINETO_ACTIVITIES, // must be the last one
};

inline std::string actToString(ActivityType t) {
  const std::array<
      std::string,
      static_cast<size_t>(ActivityType::NUM_KINETO_ACTIVITIES)>
      ActivityTypeNames = {"CPU", "XPU", "CUDA", "MTIA", "PrivateUse1"};
  return ActivityTypeNames[static_cast<int>(t)];
}

enum class C10_API_ENUM ProfilerState {
  Disabled = 0,
  CPU, // CPU-only profiling
  CUDA, // CPU + CUDA events
  NVTX, // only emit NVTX markers
  ITT, // only emit ITT markers
  PRIVATEUSE1, // only emit PRIVATEUSE1 markers
  KINETO, // use libkineto
  KINETO_GPU_FALLBACK, // use CUDA events when CUPTI is not available
  KINETO_PRIVATEUSE1_FALLBACK, // use PrivateUse1 events
  KINETO_ONDEMAND, // run the profiler in on-demand mode
  NUM_PROFILER_STATES, // must be the last one
};

enum class C10_API_ENUM ActiveProfilerType {
  NONE = 0,
  LEGACY,
  KINETO,
  NVTX,
  ITT,
  PRIVATEUSE1
};

struct TORCH_API ExperimentalConfig {
  ExperimentalConfig(
      std::vector<std::string> profiler_metrics = {},
      bool profiler_measure_per_kernel = false,
      bool verbose = false,
      std::vector<std::string> performance_events = {},
      bool enable_cuda_sync_events = false,
      bool adjust_profiler_step = false,
      bool disable_external_correlation = false,
      bool adjust_timestamps = false);
  explicit operator bool() const;

  std::vector<std::string> profiler_metrics;
  bool profiler_measure_per_kernel;
  bool verbose;
  /*
   * List of performance events to be profiled.
   * An empty list will disable performance event based profiling altogether.
   */
  std::vector<std::string> performance_events;
  /*
   * For CUDA profiling mode, enable adding CUDA synchronization events
   * that expose CUDA device, stream and event synchronization activities.
   * This feature is new and currently disabled by default.
   */
  bool enable_cuda_sync_events;
  /*
   * Controls whether or not timestamp adjustment for ProfilerStep and parent
   * Python events occurs after profiling. This occurs at an O(n) cost and
   * affects only the start of profiler step events.
   */
  bool adjust_profiler_step;
  /*
   * Controls whether or not external correlation is disabled. This is used to
   * lower the amount of events received by CUPTI as correlation events are
   * paired with runtime/gpu events for each kind of correlation
   */
  bool disable_external_correlation;

  /*
   * Controls whether or not timestamp adjustment occurs after profiling.
   * The purpose of this is to adjust Vulkan event timelines to align with those
   * of their parent CPU events.
   * This sometimes requires increasing CPU event durations (to fully contain
   * their child events) and delaying CPU event start times (to
   * prevent overlaps), so this should not be used unless Vulkan events are
   * being profiled and it is ok to use this modified timestamp/duration
   * information instead of the original information.
   */
  bool adjust_timestamps;
};

struct TORCH_API ProfilerConfig {
  explicit ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false,
      ExperimentalConfig experimental_config = ExperimentalConfig(),
      std::string trace_id = "");

  bool disabled() const;
  bool global() const;

  ProfilerState state;
  ExperimentalConfig experimental_config;
  bool report_input_shapes;
  bool profile_memory;
  bool with_stack;
  bool with_flops;
  bool with_modules;
  std::string trace_id;

  // For serialization
  at::IValue toIValue() const;
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
struct TORCH_API ProfilerStateBase : public c10::MemoryReportingInfoBase {
  explicit ProfilerStateBase(ProfilerConfig config);
  ProfilerStateBase(const ProfilerStateBase&) = delete;
  ProfilerStateBase(ProfilerStateBase&&) = delete;
  ProfilerStateBase& operator=(const ProfilerStateBase&) = delete;
  ProfilerStateBase& operator=(ProfilerStateBase&&) = delete;
  ~ProfilerStateBase() override;

  static ProfilerStateBase* get(bool global);
  static ProfilerStateBase* get() {
    auto* out = get(/*global=*/true);
    return out ? out : get(/*global=*/false);
  }

  static void push(std::shared_ptr<ProfilerStateBase>&& state);

  static std::shared_ptr<ProfilerStateBase> pop(bool global);
  static std::shared_ptr<ProfilerStateBase> pop() {
    auto out = pop(/*global=*/true);
    return out ? std::move(out) : pop(/*global=*/false);
  }

  const ProfilerConfig& config() const {
    return config_;
  }

  void setCallbackHandle(at::CallbackHandle handle);
  void removeCallback();

  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

  virtual ActiveProfilerType profilerType() = 0;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex state_mutex_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::CallbackHandle handle_ = 0;
};

// Note: The following are only for the active *thread local* profiler.
TORCH_API bool profilerEnabled();
TORCH_API ActiveProfilerType profilerType();
TORCH_API ProfilerConfig getProfilerConfig();

} // namespace torch::profiler::impl
