#pragma once

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
      int run_limit = -1);

  std::chrono::microseconds interval() const;
  int counter() const;
  void reset_counter();
  bool immediate() const;
  int run_limit() const;

  void set_next_run(
      std::chrono::time_point<std::chrono::steady_clock> next_run);

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
};

class FunctionScheduler {
  int _current_id = 0;
  std::atomic_bool _running = false;
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
  void runNextJob();
  std::chrono::microseconds getNextWaitTime();
  void addRun(int job_id, std::unique_ptr<Job> const& job);
  int scheduleJob(std::unique_ptr<Job> job);

 public:
  static constexpr int RUN_FOREVER = -1;

  FunctionScheduler();
  ~FunctionScheduler();

  template <typename Interval>
  int scheduleJob(
      std::function<void()> function,
      Interval interval,
      bool immediate = false,
      int run_limit = RUN_FOREVER);

  int removeJob(int id);

  void start();
  void stop();

  bool isRunning() const;
  int currentId() const;
};

// Template function must be defined in the header file
template <typename Interval>
int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    Interval interval,
    bool immediate,
    int run_limit) {
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(interval);
  auto job = std::make_unique<Job>(
      std::move(function), duration, immediate, run_limit);
  return scheduleJob(std::move(job));
}

} // namespace c10
