#include <c10/util/FunctionScheduler.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

/* Job */

Job::Job(std::function<void()> function, std::chrono::microseconds interval)
    : _function(std::move(function)), _interval(interval) {}

std::chrono::microseconds Job::interval() const {
  return _interval;
}

void Job::run() const {
  _function();
}

/* Run */

/* static */ bool Run::lt(Run a, Run b) {
  return a.time() < b.time();
}

Run::Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time) : _job_id(job_id), _time(time) {}

int Run::job_id() const {
  return _job_id;
}

std::chrono::time_point<std::chrono::steady_clock> Run::time() const {
  return _time;
}

/* FunctionScheduler */

FunctionScheduler::FunctionScheduler() = default;

FunctionScheduler::~FunctionScheduler() {
  stop();
}

std::chrono::microseconds FunctionScheduler::getNextWaitTime() {
  _next_run = _queue.top(); // TODO may be removed in the meantime?
  _queue.pop();

  while (_jobs.find(_next_run.job_id()) == _jobs.end()) {
      _next_run = _queue.top();
      _queue.pop();
  }

  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(_next_run.time() - now);
}

void FunctionScheduler::run() {
  while (_running) {
    if (_queue.empty()) {
      // TODO wait with mutex or check periodically?
      continue;
    }

    std::chrono::microseconds wait_time = getNextWaitTime();
    if (wait_time.count() > 0) {
        /* TODO
        * 1. other jobs may be scheduled with lower interval while this is waiting
        * 2. stop() may occur while waiting
        */
        std::this_thread::sleep_for(wait_time);
    }

    runNextJob();
  }
}

void FunctionScheduler::runNextJob() {
  // check if the job was removed in the meantime
  if (_jobs.find(_next_run.job_id()) != _jobs.end())
    _jobs.find(_next_run.job_id())->second.run();
}

int FunctionScheduler::id() {
  return _current_id++;
}

int FunctionScheduler::scheduleJob(Job job) {
  int job_id = id();
  _jobs.insert(std::make_pair(job_id, job));

  if (_running) {
    Run run = Run(job_id, std::chrono::steady_clock::now() + job.interval());
    _queue.push(run);
  }

  return job_id;
}

int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    std::chrono::microseconds interval) {
  return scheduleJob(Job(std::move(function), interval));
}

int FunctionScheduler::removeJob(int id) {
  return _jobs.erase(id) ? id : -1;
}

void FunctionScheduler::start() {
  auto now = std::chrono::steady_clock::now();
  for (const auto &entry : _jobs) {
    Run run = Run(entry.first, now + entry.second.interval());
    _queue.push(run);
  }

  _running = true;
  _thread = std::thread(&FunctionScheduler::run, this);
}

void FunctionScheduler::stop() {
  _running = false;
  if (_thread.joinable())
    _thread.join();
}

} // namespace c10
