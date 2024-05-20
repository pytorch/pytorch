#include <c10/util/FunctionScheduler.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

/* static */ bool Job::lt(Job a, Job b) { return a.next_run() < b.next_run(); }

Job::Job(std::function<void()> function, std::chrono::microseconds interval)
    : _function(function), _interval(interval) {}

std::chrono::time_point<std::chrono::steady_clock> Job::next_run() const {
  return _next_run;
}

std::chrono::microseconds Job::interval() const {
  return _interval;
}

int Job::id() const {
  return _id;
}

void Job::set_next_run(std::chrono::time_point<std::chrono::steady_clock> next_run) {
  _next_run = next_run;
}

void Job::set_id(int id) {
  _id = id;
}

void Job::run() {
  _function();
}

FunctionScheduler::FunctionScheduler() {}

FunctionScheduler::~FunctionScheduler() {
  stop();
}

void FunctionScheduler::runNextJob() {}

std::chrono::microseconds getNextInterval() {}

int FunctionScheduler::id() {
  return _current_id++;
}

int FunctionScheduler::scheduleJob(Job job) {
  int job_id = id();
  job.set_id(job_id);

  if (!_running) {
    _jobs.push_back(job);
  } else {
    job.set_next_run(std::chrono::steady_clock::now() + job.interval());
    _queue.insert(job);
  }

  return job_id;
}

int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    std::chrono::microseconds interval) {

  return scheduleJob(Job(function, interval));
}

int FunctionScheduler::removeJob(int id) {}

void FunctionScheduler::start() {
  while (!_jobs.empty()) {
    Job job = _jobs.back();
    _jobs.pop_back();
    job.set_next_run(std::chrono::steady_clock::now() + job.interval());
    _queue.insert(job);
  }
  _running = true;
}

void FunctionScheduler::stop() {}

} // namespace c10
