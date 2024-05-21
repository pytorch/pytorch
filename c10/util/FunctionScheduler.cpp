#include <c10/util/FunctionScheduler.h>

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

/* static */ bool Run::lt(std::shared_ptr<Run> const &a, std::shared_ptr<Run> const &b) {
  return a->time() < b->time();
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
  // We can't pop the next run instantly,
  // as it may still change while we're waiting.
  _next_run = _queue.top();

  // Finding the first run associated with an active job.
  while (_jobs.find(_next_run->job_id()) == _jobs.end()) {
    // Only pop runs associated with an invalid job.
    _queue.pop();
    _next_run = _queue.top();
  }

  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(_next_run->time() - now);
}

void FunctionScheduler::run() {
  std::unique_lock<std::mutex> lock(_mutex);

  while (_running) {
    if (_queue.empty()) {
      _cond.wait(lock);
      continue;
    }

    std::chrono::microseconds wait_time = getNextWaitTime();
    if (wait_time.count() > 0) {
      // Waiting for the next run to be ready.
      // We need to wake up if a new run is added
      // to the queue, as it may need to happen
      // before the current ´_next_run´
      if (_cond.wait_for(lock, wait_time) == std::cv_status::timeout) {
        // Lock timed out, i.e., no new run was added while we waited.
        // The run selected as next is still the correct one.
        runNextJob();
      }
    } else {
      runNextJob();
    }
  }
}

void FunctionScheduler::runNextJob() {
  // This function is always called with the mutex previously acquired.
  _queue.pop(); // Remove this run from the queue
  // TODO: check that the queue top is actually the _next_run (?)

  // Check if the job was canceled in the meantime.
  auto entry = _jobs.find(_next_run->job_id());
  if (entry != _jobs.end()) {
    entry->second->run();
    // Add a new run associated with this job to the queue
    addRun(entry->first, entry->second);
  }
}

int FunctionScheduler::id() {
  return _current_id++;
}

void FunctionScheduler::addRun(int job_id, std::unique_ptr<Job> const &job) {
  // This function is always called with the mutex previously acquired.
  auto time = std::chrono::steady_clock::now() + job->interval();
  auto run = std::make_shared<Run>(job_id, time);
  _queue.push(std::move(run));

  // Notify the thread handling run execution.
  if (_running) _cond.notify_one();
}

int FunctionScheduler::scheduleJob(std::unique_ptr<Job> job) {
  std::lock_guard<std::mutex> lock(_mutex);
  int job_id = id();

  if (_running) {
    addRun(job_id, job);
  }

  _jobs.insert(std::make_pair(job_id, std::move(job)));
  return job_id;
}

int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    std::chrono::microseconds interval) {
  auto job = std::make_unique<Job>(std::move(function), interval);
  return scheduleJob(std::move(job));
}

int FunctionScheduler::removeJob(int id) {
  std::lock_guard<std::mutex> lock(_mutex);
  // The scheduler checks if the job associated
  // with a run is valid, so, to cancel a job
  // and it's run, we just need to erase
  // it (thus making it invalid).
  return _jobs.erase(id) ? id : -1;
}

void FunctionScheduler::start() {
  std::lock_guard<std::mutex> lock(_mutex);
  auto now = std::chrono::steady_clock::now();
  for (const auto &entry : _jobs) {
    auto run = std::make_shared<Run>(entry.first, now + entry.second->interval());
    _queue.push(std::move(run));
  }

  _running = true;
  _thread = std::thread(&FunctionScheduler::run, this);
}

void FunctionScheduler::stop() {
  _running = false;
  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // exits the loop.
  _cond.notify_one();
  _thread.join();
  // TODO: clear queue
}

} // namespace c10
