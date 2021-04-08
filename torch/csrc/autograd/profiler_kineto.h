#pragma once

#include <torch/csrc/autograd/profiler_legacy.h>

// Kineto is currently available on Linux server-side
#ifdef USE_KINETO
#if !defined(__linux__) || defined(_WIN32) || defined(C10_MOBILE) || defined(__APPLE__) || defined(DISABLE_KINETO)
#undef USE_KINETO
#endif
#endif

#ifdef USE_KINETO
namespace libkineto {
class TraceActivity;
class ActivityTraceInterface;
}
#endif

namespace torch {
namespace autograd {
namespace profiler {

enum class C10_API_ENUM ActivityType {
  CPU = 0,
  CUDA, // CUDA kernels, runtime
  NUM_KINETO_ACTIVITIES, // must be the last one
};

#ifdef USE_KINETO

struct KinetoObserverContext : public at::ObserverContext {
  int64_t startUs;
  uint64_t correlationId;
  uint64_t startThreadId;
  uint64_t endThreadId;
  c10::optional<std::vector<std::vector<int64_t>>> shapes;
  int64_t sequenceNr;
  uint64_t fwdThreadId;
  uint8_t recFunScope;
  c10::optional<std::vector<std::string>> stack;
  // Extra arguments for computing op flops
  c10::optional<std::unordered_map<std::string, c10::IValue>> extraArgs;
};

struct TORCH_API KinetoEvent {
  KinetoEvent();

  uint64_t startThreadId() const {
    return start_thread_id_;
  }

  uint64_t endThreadId() const {
    return end_thread_id_;
  }

  uint8_t activityType() const {
    return activity_type_;
  }

  uint64_t fwdThreadId() const {
    return fwd_thread_id_;
  }

  bool hasShapes() const {
    return shapes_ != c10::nullopt;
  }

  const std::vector<std::vector<int64_t>>& shapes() const {
    return *shapes_;
  }

  uint64_t flops() const {
    return flops_;
  }

  int64_t sequenceNr() const {
    return sequence_nr_;
  }

  bool hasStack() const {
    return stack_ != c10::nullopt;
  }

  const std::vector<std::string>& stack() const {
    return *stack_;
  }

  uint8_t scope() const {
    return scope_;
  }

  KinetoEvent& startThreadId(uint64_t start_thread_id) {
    start_thread_id_ = start_thread_id;
    return *this;
  }

  KinetoEvent& endThreadId(uint64_t end_thread_id) {
    end_thread_id_ = end_thread_id;
    return *this;
  }

  KinetoEvent& fwdThreadId(uint64_t fwd_thread_id) {
    fwd_thread_id_ = fwd_thread_id;
    return *this;
  }

  KinetoEvent& shapes(const std::vector<std::vector<int64_t>>& shapes) {
    shapes_ = shapes;
    return *this;
  }

  KinetoEvent& flops(uint64_t flops) {
    flops_ = flops;
    return *this;
  }

  KinetoEvent& sequenceNr(int64_t sequence_nr) {
    sequence_nr_ = sequence_nr;
    return *this;
  }

  KinetoEvent& stack(const std::vector<std::string>& st) {
    stack_ = st;
    return *this;
  }

  KinetoEvent& scope(uint8_t scope) {
    scope_ = scope;
    return *this;
  }

  // Kineto fields

  KinetoEvent& activity(const libkineto::TraceActivity& activity);

  std::string name() const {
    return name_;
  }

  uint64_t deviceIndex() const {
    return device_index_;
  }

  uint64_t startUs() const {
    return start_us_;
  }

  uint64_t durationUs() const {
    return duration_us_;
  }

  uint64_t correlationId() const {
    return correlation_id_;
  }

  KinetoEvent& correlationId(uint64_t correlation_id)  {
    correlation_id_ = correlation_id;
    return *this;
  }

  uint64_t linkedCorrelationId() const {
    return linked_correlation_id_;
  }

  int64_t deviceResourceId() const {
    return device_resource_id_;
  }

  c10::DeviceType deviceType() const;

  uint64_t start_thread_id_ = 0;
  uint64_t end_thread_id_ = 0;
  uint64_t fwd_thread_id_ = 0;
  int64_t sequence_nr_ = -1;
  uint8_t scope_ = 0;

  uint8_t activity_type_;
  c10::optional<std::vector<std::vector<int64_t>>> shapes_;
  c10::optional<std::vector<std::string>> stack_;
  uint64_t flops_ = 0;

  std::string name_;
  uint64_t device_index_ = 0;
  uint64_t start_us_ = 0;
  uint64_t duration_us_ = 0;
  uint64_t correlation_id_ = 0;
  uint64_t linked_correlation_id_ = 0;
  int64_t device_resource_id_ = 0;
};

// Consolidating events returned directly from Kineto
// with events manually created by us (e.g. start/stop marks,
// memory allocation events)
struct TORCH_API ProfilerResult {
  ProfilerResult(
      std::vector<KinetoEvent> events,
      thread_event_lists legacy_events,
      std::unique_ptr<libkineto::ActivityTraceInterface> trace);
  ~ProfilerResult();

  const std::vector<KinetoEvent>& events() const {
    return events_;
  }

  const thread_event_lists& legacy_events() const {
    return legacy_events_;
  }

  void save(const std::string& path);

 private:
  bool saved_ = false;
  std::vector<KinetoEvent> events_;
  thread_event_lists legacy_events_;
  std::unique_ptr<libkineto::ActivityTraceInterface> trace_;
};

TORCH_API void enableProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);

TORCH_API std::unique_ptr<ProfilerResult> disableProfiler();

TORCH_API void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);
#endif // USE_KINETO

} // namespace profiler
}} // namespace torch::autograd
