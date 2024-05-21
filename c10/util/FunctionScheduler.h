#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <queue>
#include <thread>
#include <vector>
#include <unordered_map>

namespace c10 {

class Job {
  std::function<void()> _function;
  std::chrono::microseconds _interval;

 public:
  Job(std::function<void()> function, std::chrono::microseconds interval);

  std::chrono::time_point<std::chrono::steady_clock> next_run() const;
  std::chrono::microseconds interval() const;

  void set_next_run(
      std::chrono::time_point<std::chrono::steady_clock> next_run);

  void run() const;
};

class Run {
  int _job_id;
  std::chrono::time_point<std::chrono::steady_clock>  _time;

 public:
  static bool lt(Run a, Run b);

  Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time);

  int job_id() const;
  std::chrono::time_point<std::chrono::steady_clock> time() const;

};

class FunctionScheduler {
  int _current_id = 0;
  std::atomic_bool _running = false;
  std::priority_queue<Run, std::vector<Run>, decltype(&Run::lt)> _queue;
  std::unordered_map<int, Job> _jobs;
  Run _next_run = Run(-1, std::chrono::steady_clock::now()); // TODO remove this after changing to unique_ptr
  std::thread _thread;

  void run();
  void runNextJob();
  std::chrono::microseconds getNextWaitTime();
  int id();

 public:
  FunctionScheduler();
  ~FunctionScheduler();

  int scheduleJob(Job job);
  int scheduleJob(
      std::function<void()> function,
      std::chrono::microseconds interval);

  int removeJob(int id);

  void start();
  void stop();
};

} // namespace c10
