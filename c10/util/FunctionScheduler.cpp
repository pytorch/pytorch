#include <c10/util/FunctionScheduler.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

/* Job */

bool Job::lt(const Job& a, const Job& b) {
  return a.next_run() < b.next_run();
}

Job::Job(std::function<void()> function, std::chrono::microseconds interval)
    : _function(std::move(function)), _interval(interval) {}

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

void FunctionScheduler::run() {
  while (_running) {
    if (_queue.empty()) {
      // TODO wait with mutex or check periodically?
      continue;
    }
    runNextJob();
  }
}

void FunctionScheduler::runNextJob() {
  Job next_job = *_queue.begin(); // TODO may be removed in the meantime?
  std::chrono::microseconds next_interval = getJobInterval(next_job);
  if (next_interval.count() > 0) {
    /* TODO
     * 1. other jobs may be scheduled with lower interval while this is waiting
     * 2. stop() may occur while waiting
     */
    std::this_thread::sleep_for(next_interval);
  }

  // check if the job was removed in the meantime
  if (_queue.begin()->id() == next_job.id())
    next_job.run();
}

std::chrono::microseconds FunctionScheduler::getJobInterval(Job& job) const {
  auto interval = job.next_run() - std::chrono::steady_clock::now();
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
  return scheduleJob(Job(std::move(function), interval));
}

int FunctionScheduler::removeJob(int id) {
  if (_running) {
    for (auto it = _queue.begin(); it != _queue.end();) {
      if (it->id() == id) {
        _queue.erase(it);
        return id;
      }
      it++;
    }
    return -1;
  } else {
    const auto end = _jobs.end();
    const auto updated_end = _jobs.erase(
        std::remove_if(
            _jobs.begin(),
            _jobs.end(),
            [id](const Job& job) { return job.id() == id; }),
        end);
    return updated_end == end ? -1 : id;
  }
}

void FunctionScheduler::start() {
  auto now = std::chrono::steady_clock::now();
  while (!_jobs.empty()) {
    Job job = _jobs.back();
    _jobs.pop_back();
    job.set_next_run(now + job.interval());
    _queue.insert(job);
  }

  _running = true;
  _thread = std::thread(&FunctionScheduler::run, this);
}

void FunctionScheduler::stop() {
  _running = false;
  if (_thread.joinable())
    _thread.join();

  // save unfinished jobs
  _jobs.insert(_jobs.end(), _queue.begin(), _queue.end());
  _queue.clear();
}

} // namespace c10
