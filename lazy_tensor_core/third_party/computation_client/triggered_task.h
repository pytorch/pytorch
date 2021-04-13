#ifndef COMPUTATION_CLIENT_TRIGGERED_TASK_H_
#define COMPUTATION_CLIENT_TRIGGERED_TASK_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lazy_tensors {
namespace util {

// Wraps a function which should be run many times upon user activations.
class TriggeredTask {
 public:
  // Note that if num_threads > 1, the function will be run concurrently from
  // multiple threads, so it will have to be thread safe. This condition does
  // not apply if num_threads is 1.
  TriggeredTask(std::function<void()> function, size_t num_threads);

  // Stops the background thread and waits for it to complete.
  void Stop();

  // Triggers a function run. If the function is already running, it will run
  // again immediately after it completes. Returns tthe value of thte run-ID the
  // caller should eventually wait with the WaitForRun() API, to be sure that a
  // full function run happened after its Activate() call.
  size_t Activate();

  // Wait until a run-ID returned by the Activate() API completed. Returns the
  // value of the current run-ID. If such value or less or equal to run_id, the
  // wait did not complete successfully.
  size_t WaitForRun(size_t run_id);

 private:
  // Function implementing the main thread loop running the user function.
  void Runner();

  std::function<void()> function_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable run_cv_;
  size_t run_id_ = 0;
  size_t run_waiters_ = 0;
  size_t running_ = 0;
  bool activated_ = false;
  bool stopped_ = false;
  std::vector<std::unique_ptr<std::thread>> threads_;
};

}  // namespace util
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_TRIGGERED_TASK_H_
