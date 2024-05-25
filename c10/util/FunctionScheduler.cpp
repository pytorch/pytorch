#include <c10/util/FunctionScheduler.h>

#include <iostream>

namespace c10 {

/* Job */

Job::Job(
    std::function<void()> function,
    std::chrono::microseconds interval,
    bool immediate,
    int run_limit)
    : _function(std::move(function)),
      _interval(interval),
      _immediate(immediate),
      _run_limit(run_limit) {}

void Job::run() {
  _counter++;
  try {
    _function();
  } catch (const std::exception& e) {
    std::cerr << "Job failed: " << e.what() << std::endl;
  }
}

/* Run */

Run::Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time)
    : _job_id(job_id), _time(time) {}

/* FunctionScheduler */

FunctionScheduler::FunctionScheduler() = default;

FunctionScheduler::~FunctionScheduler() {
  stop();
}

std::chrono::microseconds FunctionScheduler::getNextWaitTime() {
  // We can't pop the next run instantly,
  // as it may still change while we're waiting.
  _next_run = _queue.front();

  // Finding the first run associated with an active job.
  auto entry = _jobs.find(_next_run.job_id());
  while (!validEntry(entry)) {
    // Only pop runs associated with an invalid job.
    std::pop_heap(_queue.begin(), _queue.end(), Run::gt);
    _queue.pop_back();

    if (_queue.empty())
      return std::chrono::microseconds(-1);

    _next_run = _queue.front();
    entry = _jobs.find(_next_run.job_id());
  }

  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
      _next_run.time() - now);
}

void FunctionScheduler::run() {
  std::unique_lock<std::mutex> lock(_mutex);

  while (_running) {
    if (_queue.empty() || _paused) {
      _cond.wait(lock);
      continue;
    }

    std::chrono::microseconds wait_time = getNextWaitTime();
    // Check again if queue is empty after pops
    if (_queue.empty())
      continue;

    if (wait_time.count() > 0) {
      // Waiting for the next run to be ready.
      // We need to wake up if a new run is added
      // to the queue, as it may need to happen
      // before the current ´_next_run´. We also
      // need to wake up if ´_paused´ changes, to
      // pause execution.
      if (_cond.wait_for(lock, wait_time) == std::cv_status::timeout) {
        // Lock timed out, i.e., nothing happened while we waited.
        // The run selected as next is still the correct one and we
        // aren't paused.
        runNextJob(lock);
      }
    } else {
      runNextJob(lock);
    }
  }
}

void FunctionScheduler::runNextJob(const std::unique_lock<std::mutex>& lock) {
  // This function is always called with the mutex previously acquired.
  TORCH_INTERNAL_ASSERT(lock.owns_lock(), "Mutex not acquired");
  TORCH_INTERNAL_ASSERT(
      _next_run == _queue.front(), "Next run does not match queue top.");

 // Remove this run from the queue
  std::pop_heap(_queue.begin(), _queue.end(), Run::gt);
  _queue.pop_back();

  // Check if the job was canceled in the meantime.
  auto entry = _jobs.find(_next_run.job_id());
  if (validEntry(entry)) {
    entry->second->run();
    // Add a new run associated with this job to the queue
    addRun(lock, entry->first, entry->second);
  }
}

bool FunctionScheduler::validEntry(
    const std::unordered_map<int, std::unique_ptr<Job>>::iterator& entry) {
  return entry != _jobs.end() &&
      entry->second->counter() != entry->second->run_limit();
}

void FunctionScheduler::addRun(
    const std::unique_lock<std::mutex>& lock,
    int job_id,
    const std::unique_ptr<Job>& job) {
  // This function is always called with the mutex previously acquired.
  TORCH_INTERNAL_ASSERT(lock.owns_lock(), "Mutex not acquired");

  auto interval = job->interval();
  if (job->immediate() && job->counter() == 0)
    interval = std::chrono::microseconds(0);

  auto time = std::chrono::steady_clock::now() + interval;
  Run run = Run(job_id, time);

  _queue.push_back(run);
  std::push_heap(_queue.begin(), _queue.end(), Run::gt);
}

int FunctionScheduler::scheduleJob(std::unique_ptr<Job> job) {
  std::unique_lock<std::mutex> lock(_mutex);
  int job_id = id();

  if (_running) {
    addRun(lock, job_id, job);
    // Notify the thread handling run execution.
    _cond.notify_one();
  }
  _jobs.insert(std::make_pair(job_id, std::move(job)));

  return job_id;
}

int FunctionScheduler::removeJob(int id) {
  std::lock_guard<std::mutex> lock(_mutex);
  // The scheduler checks if the job associated
  // with a run is valid, so, to cancel a job
  // and it's run, we just need to erase
  // it (thus making it invalid).
  return _jobs.erase(id) ? id : -1;
}

int FunctionScheduler::start() {
  if (_running || _paused)
    return -1;

  std::unique_lock<std::mutex> lock(_mutex);
  for (const auto& entry : _jobs) {
    addRun(lock, entry.first, entry.second);
  }

  _running = true;
  _paused = false;
  _thread = std::thread(&FunctionScheduler::run, this);
  return 1;
}

int FunctionScheduler::stop() {
  if (!_running)
    return -1;

  _running = false;
  _paused = false;
  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // exits the loop.
  _cond.notify_one();
  if (_thread.joinable()) {
    _thread.join();
  }

  // clear queue
  _queue.clear();

  // reset counters
  for (const auto& entry : _jobs) {
    entry.second->reset_counter();
  }
  return 1;
}

int FunctionScheduler::pause() {
  if (_paused || !_running)
    return -1;

  _paused_time = std::chrono::steady_clock::now();
  _paused = true;
  return 1;
}

int FunctionScheduler::resume() {
  if (!_paused)
    return -1;

  std::lock_guard<std::mutex> lock(_mutex);

  // Since we're shifting the time of all elements by the same amount
  // the min-heap is still valid, no need to rebuild it.
  auto diff = std::chrono::steady_clock::now() - _paused_time;
  for (auto& run : _queue) {
    run.set_time(run.time() + diff);
  }

  _paused = false;

  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // continues execution.
  _cond.notify_one();

  return 1;
}

} // namespace c10
