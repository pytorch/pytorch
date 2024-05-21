#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <set>
#include <thread>
#include <vector>

namespace c10 {

class Job {
  int _id = -1;
  std::function<void()> _function;
  std::chrono::microseconds _interval;
  std::chrono::time_point<std::chrono::steady_clock> _next_run;

 public:
  static bool lt(const Job& a, const Job& b);

  Job(std::function<void()> function, std::chrono::microseconds interval);

  int id() const;
  std::chrono::time_point<std::chrono::steady_clock> next_run() const;
  std::chrono::microseconds interval() const;

  void set_id(int id);
  void set_next_run(
      std::chrono::time_point<std::chrono::steady_clock> next_run);

  void run() const;
};

class FunctionScheduler {
  int _current_id = 0;
  std::atomic_bool _running = false;
  std::multiset<Job, decltype(&Job::lt)> _queue;
  std::vector<Job> _jobs;
  std::thread _thread;

  void run();
  void runNextJob();
  std::chrono::microseconds getJobInterval(Job& job) const;
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
