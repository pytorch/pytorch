#pragma once

#include <c10/util/Exception.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace c10 {

/**
 * Represents a function that runs periodically.
 */
class C10_API Job {
  std::function<void()> _function;
  std::chrono::microseconds _interval;
  int _counter = 0;
  int _run_limit;
  bool _immediate;

 public:
  Job(std::function<void()> function,
      std::chrono::microseconds interval,
      bool immediate = false,
      int run_limit = -1); // -1 = FunctionScheduler::RUN_FOREVER

  std::chrono::microseconds interval() const {
    return _interval;
  }

  int counter() const {
    return _counter;
  }

  void reset_counter() {
    _counter = 0;
  }

  bool immediate() const {
    return _immediate;
  }

  int run_limit() const {
    return _run_limit;
  }

  void run();
};

/**
 * Represents a concrete run, i.e, a job that
 * will be executed at a specific time.
 */
class C10_API Run {
  int _job_id = -1;
  std::chrono::time_point<std::chrono::steady_clock> _time;

 public:
  static bool gt(const Run& a, const Run& b) {
    return a.time() > b.time();
  }

  Run() = default;

  Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time);

  int job_id() const {
    return _job_id;
  }

  std::chrono::time_point<std::chrono::steady_clock> time() const {
    return _time;
  }

  void set_time(std::chrono::time_point<std::chrono::steady_clock> time) {
    _time = time;
  }

  bool operator==(const Run& other) {
    return _job_id == other.job_id() && _time == other.time();
  }
};

/**
 * Schedule a function to run periodically.
 * Example:
 * bool ran = false;
 * std::function<void()> function = [&ran]() { ran = true; };
 * std::chrono::milliseconds interval(10);
 *
 * c10::FunctionScheduler fs;
 * fs.scheduleJob(function, interval);
 * fs.start();
 * std::this_thread::sleep_for(std::chrono::milliseconds(2));
 * // ran == false
 *
 * std::this_thread::sleep_for(std::chrono::milliseconds(12));
 * fs.stop();
 * // ran == true
 *
 */
class C10_API FunctionScheduler {
 public:
  static constexpr int RUN_FOREVER = -1;

 private:
  // The id to be attributed to a new job.
  int _current_id = 0;

  // FunctionScheduler state.
  std::atomic_bool _running = false;
  std::atomic_bool _paused = false;
  std::chrono::time_point<std::chrono::steady_clock> _paused_time;
  // Tracks how many operations (sheduleJob/removeJob) are pending
  std::atomic_int _dirty = 0;

  // Runs, sorted by wait time until execution.
  std::vector<Run> _queue;

  // Current active jobs.
  std::unordered_map<int, Job> _jobs;

  // Run selected to be executed next
  Run _next_run;

  // The thread running the run execution loop
  std::thread _thread;

  // Synchronization variables.
  std::mutex _mutex;
  std::condition_variable _cond;

  // Returns a new job id, updating _current_id.
  int id() {
    return _current_id++;
  }

  // Main run execution loop function.
  void run();

  // Executes _next_run
  void runNextJob(const std::unique_lock<std::mutex>& lock);

  // Selects the next run to be executed and returns
  // the wait time until execution.
  std::chrono::microseconds getNextWaitTime();

  // Used when adding a run while the scheduler is not running,
  // so there's no need to lock.
  void addRun(int job_id, const Job& job);

  // Used when adding a run while the scheduler is running,
  // must be called with a lock acquired.
  void addRun(
      const std::unique_lock<std::mutex>& lock,
      int job_id,
      const Job& job);

  // Registers a new run. Should not be used directly,
  // use addRun instead.
  void addRunInternal(int job_id, const Job& job);

  // Registers a new job.
  int scheduleJob(const Job& job);

  // Checks if a job is still valid.
  bool validEntry(const std::unordered_map<int, Job>::iterator& entry);

 public:
  FunctionScheduler();
  ~FunctionScheduler();

  // Registers a new job that runs `function` every `interval`.
  // If `immediate` is false, `function` runs for the
  // first time only after `interval` has passed.
  // `run_limit` is the max times `function` will run.
  // Use FunctionScheduler::RUN_FOREVER for no limit.
  template <typename Rep, typename Period>
  int scheduleJob(
      std::function<void()> function,
      std::chrono::duration<Rep, Period> interval,
      bool immediate = false,
      int run_limit = RUN_FOREVER) {
    TORCH_CHECK(function != nullptr, "Job function can't be null.");
    TORCH_CHECK(interval.count() >= 0, "Job interval must be positive.");
    TORCH_CHECK(
        run_limit > 0 || run_limit == RUN_FOREVER,
        "Job run limit must be greater than 0 or FunctionScheduler::RUN_FOREVER (",
        RUN_FOREVER,
        ").");

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(interval);
    Job job = Job(std::move(function), duration, immediate, run_limit);
    return scheduleJob(job);
  }

  // Removes the job registered with `id` and returns it.
  // Returns false if a job registered with `id` doesn't exist.
  bool removeJob(int id);

  // Starts the FunctionScheduler.
  // Returns false if the Scheduler is already running.
  bool start();

  // Stops the FunctionScheduler
  // and resets all jobs (_run_count = 0).
  // Returns false if the Scheduler is already stopped.
  bool stop();

  // Pauses FunctionScheduler execution.
  // Returns false if the Scheduler is already
  // paused or not running.
  bool pause();

  // Resumes FunctionScheduler execution.
  // Returns false if the Scheduler is already
  // unpaused or not running.
  bool resume();

  bool isRunning() const {
    return _running;
  }
  int currentId() const {
    return _current_id;
  }
};

} // namespace c10
