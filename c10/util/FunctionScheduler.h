#pragma once

#include <c10/macros/Macros.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>

#define RUN_FOREVER -1

namespace c10 {

// Represents a function that runs
// periodically.
class Job {
  std::function<void()> _function;
  std::chrono::microseconds _interval;
  int _counter = 0;
  int _run_limit;
  bool _immediate;

 public:
  Job(std::function<void()> function,
      std::chrono::microseconds interval,
      bool immediate = false,
      int run_limit = RUN_FOREVER);

  std::chrono::microseconds interval() const;
  int counter() const;
  void reset_counter();
  bool immediate() const;
  int run_limit() const;

  void run();
};

// Represents a concrete run, i.e,
// a job that will be executed at
// a specific time.
class Run {
  int _job_id;
  std::chrono::time_point<std::chrono::steady_clock> _time;

 public:
  static bool gt(std::shared_ptr<Run> const& a, std::shared_ptr<Run> const& b);

  Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time);

  int job_id() const;
  std::chrono::time_point<std::chrono::steady_clock> time() const;

  void set_time(std::chrono::time_point<std::chrono::steady_clock> time);
};

class FunctionScheduler {
  int _current_id = 0;
  std::atomic_bool _running = false;
  std::atomic_bool _paused = false;
  std::chrono::time_point<std::chrono::steady_clock> _paused_time;
  std::priority_queue<
      std::shared_ptr<Run>,
      std::vector<std::shared_ptr<Run>>,
      decltype(&Run::gt)>
      _queue;
  std::unordered_map<int, std::unique_ptr<Job>> _jobs;
  std::shared_ptr<Run> _next_run;
  std::thread _thread;
  std::mutex _mutex;
  std::condition_variable _cond;

  int id();
  void run();
  void runNextJob(const std::unique_lock<std::mutex>& lock);
  std::chrono::microseconds getNextWaitTime();
  void addRun(const std::unique_lock<std::mutex>& lock, int job_id, std::unique_ptr<Job> const& job);
  int scheduleJob(std::unique_ptr<Job> job);
  bool validEntry(const std::unordered_map<int, std::unique_ptr<Job>>::iterator& entry);

 public:
  FunctionScheduler();
  ~FunctionScheduler();

  template <typename Rep, typename Period>
  int scheduleJob(
      std::function<void()> function,
      std::chrono::duration<Rep, Period> interval,
      bool immediate = false,
      int run_limit = RUN_FOREVER);

  int removeJob(int id);

  void start();
  void stop();
  void pause();
  void resume();

  bool isRunning() const;
  int currentId() const;
};

// Template function must be defined in the header file
template <typename Rep, typename Period>
int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    std::chrono::duration<Rep, Period> interval,
    bool immediate,
    int run_limit) {
  TORCH_CHECK(function != nullptr, "Job function can't be null.");
  TORCH_CHECK(interval > 0, "Job interval must be greater than 0.");
  TORCH_CHECK(run_limit > 0 || run_limit == RUN_FOREVER, "Job run limit must be greater than 0 or " + std::to_string(RUN_FOREVER) + ".");

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(interval);
  auto job = std::make_unique<Job>(
      std::move(function), duration, immediate, run_limit);
  return scheduleJob(std::move(job));
}

} // namespace c10
