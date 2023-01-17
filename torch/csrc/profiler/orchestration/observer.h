#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/Export.h>

#include <utility>

namespace torch {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
enum class C10_API_ENUM ActivityType {
  CPU = 0,
  CUDA, // CUDA kernels, runtime
  NUM_KINETO_ACTIVITIES, // must be the last one
};

enum class C10_API_ENUM ProfilerState {
  Disabled = 0,
  CPU, // CPU-only profiling
  CUDA, // CPU + CUDA events
  NVTX, // only emit NVTX markers
  ITT, // only emit ITT markers
  KINETO, // use libkineto
  KINETO_GPU_FALLBACK, // use CUDA events when CUPTI is not available
  KINETO_ONDEMAND, // run the profiler in on-demand mode
  NUM_PROFILER_STATES, // must be the last one
};

enum class C10_API_ENUM ActiveProfilerType {
  NONE = 0,
  LEGACY,
  KINETO,
  NVTX,
  ITT
};

struct TORCH_API ExperimentalConfig {
  ExperimentalConfig(
      std::vector<std::string> profiler_metrics = {},
      bool profiler_measure_per_kernel = false,
      bool verbose = false,
      std::vector<std::string> performance_events = {},
      bool adjust_timestamps = false);
  ~ExperimentalConfig() = default;
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
   * Controls whether or not timestamp adjustment occurs after profiling.
   * The purpose of this is to adjust Vulkan event timelines to align with those
   * of their parent CPU events.
   * This sometimes requires increasing CPU event durations (to fully contain
   * their child events) and delaying CPU event start times (to
   * prevent overlaps), so this should not be used unless Vulkan events are
   * being profiled and it is ok to use this modified timestamp/duration
   * information instead of the the original information.
   */
  bool adjust_timestamps;
};

struct TORCH_API ProfilerConfig {
  ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false,
      ExperimentalConfig experimental_config = ExperimentalConfig());
  ~ProfilerConfig() = default;

  bool disabled() const;
  bool global() const;

  ProfilerState state;
  ExperimentalConfig experimental_config;
  bool report_input_shapes;
  bool profile_memory;
  bool with_stack;
  bool with_flops;
  bool with_modules;

  // For serialization
  at::IValue toIValue() const;
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
struct TORCH_API ProfilerStateBase : public c10::MemoryReportingInfoBase {
  explicit ProfilerStateBase(const ProfilerConfig& config);
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

} // namespace impl
} // namespace profiler
} // namespace torch
