#pragma once

#include <string>
#include <vector>

#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {
struct Result;
namespace kineto {
struct ActivityTraceWrapper;
} // namespace kineto
} // namespace impl
} // namespace profiler
namespace autograd {
namespace profiler {
using experimental_event_t = std::shared_ptr<torch::profiler::impl::Result>;

struct TORCH_API KinetoEvent {
  KinetoEvent(
      std::shared_ptr<const torch::profiler::impl::Result>,
      const bool verbose);

  uint64_t startThreadId() const;
  uint64_t endThreadId() const;
  uint8_t activityType() const;
  uint64_t fwdThreadId() const;
  bool hasShapes() const;
  const c10::ArrayRef<std::vector<int64_t>> shapes() const;
  bool hasTypes() const;
  const c10::ArrayRef<std::string> dtypes() const;
  uint64_t flops() const;
  int64_t sequenceNr() const;
  bool hasStack() const;
  const c10::ArrayRef<std::string> stack() const;
  uint8_t scope() const;
  bool hasModuleHierarchy() const;
  const c10::ArrayRef<std::string> moduleHierarchy() const;
  int64_t debugHandle() const;
  std::string name() const;
  c10::DeviceType deviceType() const;
  uint8_t deviceIndex() const;
  int64_t nBytes() const;
  uint64_t startUs() const;
  uint64_t durationUs() const;
  bool isAsync() const;
  uint64_t correlationId() const;
  uint64_t linkedCorrelationId() const;
  int64_t deviceResourceId() const;
  std::string backend() const;
  bool isPythonFunction() const;
  int64_t cudaElapsedUs() const;
  void getPerfEventCounters(torch::profiler::perf_counters_t&) const;

 private:
  torch::profiler::impl::ProfilerEventStub fallbackStart() const;
  torch::profiler::impl::ProfilerEventStub fallbackEnd() const;

  std::shared_ptr<const torch::profiler::impl::Result> result_;
  std::vector<std::string> python_stack_;

  // Copy fields from result so we can return ArrayRefs.
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<std::string> dtypes_;
};

// Consolidating events returned directly from Kineto
// with events manually created by us (e.g. start/stop marks,
// memory allocation events)
struct TORCH_API ProfilerResult {
  ProfilerResult();
  ProfilerResult(
      uint64_t start_time,
      std::vector<KinetoEvent> events,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&&
          trace,
      std::vector<experimental_event_t>&& event_tree);
  ~ProfilerResult();

  uint64_t trace_start_us() const {
    return trace_start_us_;
  }

  const std::vector<KinetoEvent>& events() const {
    return events_;
  }

  const std::vector<experimental_event_t>& event_tree() const {
    return event_tree_;
  }

  void save(const std::string& path);

 private:
  uint64_t trace_start_us_ = 0;
  std::vector<KinetoEvent> events_;
  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper> trace_;
  std::vector<experimental_event_t> event_tree_;
};

/*
 * This API is used by backends to record latency of events that
 * happened in the backend but were not visible to pytorch runtime.
 * For example, if part of the model is lowered to a dsp backend, then
 * the execution of that part of the model is delegated to the backend.
 * When backend finishes execution it has an option to provide profiling
 * information (latency only at th emoment) corresponding to different operators
 * that were executed in the backend.
 * When such events are recorded by backend using this API, the event
 * records will be collected by active kineto profiler. If no kineto profiler
 * is active then the event is ignored.
 * This provides us with a way to generate all the profiling information
 * for a model regardless of where model (or part of it) executed.
 * @param start_time_us: start time in us of the event
 * @param end_time_us: end time in us of the event
 * @param debug_handle: debug handle to correlate this event/op with
 * model level module/source information
 * @param scope: scope of the event, e.g. LITE_INTERPRETER, RECORD_FN etc.
 * @param event_name: name of the event, e.g. op name
 * @param backend_name: name of the backend where the event took place.
 */
TORCH_API void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name);

TORCH_API void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes = {});

/*
 * Same as enableProfiler but with callback to do post-processing of
 * KinetoEvents.
 * enableProfilerWithEventPostProcess enables profiler to capture
 * specified activities, with specified RecordFunction scope, if any.
 * Additionally, it takes a functor that does in-place post processing of
 * events, e.g. populate stack trace or module hierarchy information lazily
 * using debug_handle.
 * Example usage is with lite interpreter that has recording scope of
 * LITE_INTERPRETER. In this case lite interpreter runtime, records debug
 * handles in RecordFunction, along with other information. Debug handles are
 * eventually passed down to KinetoEvent and recorded as part of the event.
 * KinetoEdgeCPUProfiler, in torch/csrc/jit/mobile/profiler_edge.cpp, enables
 * profiler using post-processing callback, via
 * enableProfilerWithEventPostProcess, that takes these debug handles and
 * generates stack trace and module hierarchy information, once profiling is
 * done.
 */
using post_process_t = std::function<void(
    /*debug_handle */ int64_t,
    /*jit_stack    */ std::vector<std::string>&,
    /*jit_modules  */ std::vector<std::string>&)>;
TORCH_API void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    post_process_t&& cb,
    const std::unordered_set<at::RecordScope>& scopes = {});

TORCH_API std::unique_ptr<ProfilerResult> disableProfiler();

TORCH_API void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

} // namespace profiler
} // namespace autograd

namespace profiler {
namespace impl {

// Experimental.
TORCH_API void _reportVulkanEventToProfiler(vulkan_id_t id);

} // namespace impl
} // namespace profiler

} // namespace torch
