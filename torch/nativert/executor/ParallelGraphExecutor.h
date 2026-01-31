#pragma once

#include <c10/util/Semaphore.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/SessionState.h>
#include <thread>

namespace moodycamel {
struct ProducerToken;
struct ConsumerToken;
struct ConcurrentQueueDefaultTraits;
template <typename T, typename Traits>
class ConcurrentQueue;
} // namespace moodycamel

namespace torch::nativert {

/**
 * Synchronizes profiler state between main thread and child thread.
 *
 * This function checks if the main thread's profiler state has changed
 * and enables/disables profiling in the current child thread accordingly.
 * This allows worker threads in a thread pool to participate in Kineto tracing
 * when profiling is dynamically enabled/disabled.
 *
 * @param profilerEnabledInThisThread Current profiler state for this thread.
 *        Will be updated to reflect the new state after synchronization.
 */
inline void syncProfilerStateFromMainThread(bool& profilerEnabledInThisThread) {
  bool mainThreadProfiling =
      torch::autograd::profiler::isProfilerEnabledInMainThread();

  if (mainThreadProfiling != profilerEnabledInThisThread) {
    if (mainThreadProfiling) {
      torch::autograd::profiler::enableProfilerInChildThread();
    } else {
      torch::autograd::profiler::disableProfilerInChildThread();
    }
    profilerEnabledInThisThread = mainThreadProfiling;
  }
}

class ThreadPoolExecutor;

typedef std::function<void()> Work;

struct WorkUnit {
  const Node* node;
  OpKernel* kernel;
  std::vector<WorkUnit*> users;
  void run(ThreadPoolExecutor* executor, SessionState* sessionState);
};

class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor();
  ~ThreadPoolExecutor();
  ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
  ThreadPoolExecutor& operator=(ThreadPoolExecutor const&) = delete;
  ThreadPoolExecutor(ThreadPoolExecutor&&) = delete;
  ThreadPoolExecutor& operator=(ThreadPoolExecutor&&) = delete;

  void run(SessionState& session, const std::vector<WorkUnit*>& roots);

  void start(int32_t numThreads);
  void stop();

  // execute unit on the current thread
  // NOTE: children can still be offloaded to other threads
  C10_ALWAYS_INLINE void execute_inline(SessionState* session, WorkUnit* unit);

  void add(SessionState* session, WorkUnit* unit);
  void add(
      SessionState* session,
      std::vector<WorkUnit*>::const_iterator begin,
      const std::vector<WorkUnit*>::const_iterator& end);

  C10_ALWAYS_INLINE moodycamel::ProducerToken& ptok();
  C10_ALWAYS_INLINE moodycamel::ConsumerToken& ctok();

 private:
  void loop();

  std::atomic_bool stopped_{false};

  std::unique_ptr<c10::Semaphore> sem_{std::make_unique<c10::Semaphore>()};

  std::unique_ptr<moodycamel::ConcurrentQueue<
      Work,
      moodycamel::ConcurrentQueueDefaultTraits>>
      work_;
  std::vector<std::thread> threads_;
};

class ParallelGraphExecutor : public GraphExecutorBase {
 public:
  ParallelGraphExecutor(
      const Graph& graph,
      std::vector<std::unique_ptr<OpKernel>> nodeKernels,
      const ExecutorConfig& executorConfig);

  std::vector<c10::IValue> execute(
      ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) override;

  std::vector<c10::IValue> executeWithPrefilledFrame(
      ExecutionFrame& frame) override;

 private:
  ThreadPoolExecutor executor_;

  std::vector<WorkUnit*> inputWorkUnits_;
  c10::FastMap<const Node*, WorkUnit*> nodeToWorkUnit_;
  std::vector<WorkUnit> workUnits_;

  const Graph& graph_;
  c10::FastMap<const Node*, copyable_atomic<std::uint_fast32_t>> producers_;
};

} // namespace torch::nativert
