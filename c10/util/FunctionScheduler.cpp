#include "./FunctionScheduler.h"
#include <c10/util/FunctionScheduler.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

/* Job */

bool Job::lt(Job a, Job b) {
  return a.next_run() < b.next_run();
}

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

void Job::set_next_run(
    std::chrono::time_point<std::chrono::steady_clock> next_run) {
  _next_run = next_run;
}

void Job::set_id(int id) {
  _id = id;
}

void Job::run() const {
  _function();
}

/* FunctionScheduler */

FunctionScheduler::FunctionScheduler() = default;

FunctionScheduler::~FunctionScheduler() {
  stop();
}

void FunctionScheduler::runNextJob() {
  if (_queue.empty())
    return;

  if (getNextInterval().count() < 0)
    _queue.begin()->run();
}

std::chrono::microseconds FunctionScheduler::getNextInterval() const {
  if (_queue.empty())
    return std::chrono::microseconds(999);

  auto interval = _queue.begin()->next_run() - std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(interval);
}

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

int FunctionScheduler::removeJob(int id) {
  // TODO make sure job doesn't start while removing it?
  _running = false;

  // remove from _queue
  for (auto it = _queue.begin(); it != _queue.end();) {
    if (it->id() == id)
      it = _queue.erase(it);
    else
      it++;
  }
  _running = true;

  // remove from _jobs
  _jobs.erase(
      std::remove_if(
          _jobs.begin(),
          _jobs.end(),
          [id](const Job& job) { return job.id() == id; }),
      _jobs.end());
}

void FunctionScheduler::start() {
  while (!_jobs.empty()) {
    Job job = _jobs.back();
    _jobs.pop_back();
    job.set_next_run(std::chrono::steady_clock::now() + job.interval());
    _queue.insert(job);
  }
  _running = true;
}

void FunctionScheduler::stop() {
  _running = false;
  _jobs.insert(_jobs.end(), _queue.begin(), _queue.end());
  _queue.clear();
}

} // namespace c10
