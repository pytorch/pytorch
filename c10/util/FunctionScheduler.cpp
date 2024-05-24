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

std::chrono::microseconds Job::interval() const {
  return _interval;
}

int Job::counter() const {
  return _counter;
}

void Job::reset_counter() {
  _counter = 0;
}

bool Job::immediate() const {
  return _immediate;
}

int Job::run_limit() const {
  return _run_limit;
}

void Job::run() {
  _counter++;
  try {
    _function();
  } catch (const std::exception& e) {
    std::cerr << "Job failed: " << e.what() << std::endl;
  }
}

/* Run */

/* static */ bool Run::gt(
    const std::shared_ptr<Run>& a,
    const std::shared_ptr<Run>& b) {
  return a->time() > b->time();
}

Run::Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time)
    : _job_id(job_id), _time(time) {}

int Run::job_id() const {
  return _job_id;
}

std::chrono::time_point<std::chrono::steady_clock> Run::time() const {
  return _time;
}

void Run::set_time(std::chrono::time_point<std::chrono::steady_clock> time) {
  _time = time;
}

/* FunctionScheduler */

FunctionScheduler::FunctionScheduler() : _queue(&Run::gt){};

FunctionScheduler::~FunctionScheduler() {
  stop();
}

std::chrono::microseconds FunctionScheduler::getNextWaitTime() {
  // We can't pop the next run instantly,
  // as it may still change while we're waiting.
  _next_run = _queue.top();

  // Finding the first run associated with an active job.
  auto entry = _jobs.find(_next_run->job_id());
  while (!validEntry(entry)) {
    // Only pop runs associated with an invalid job.
    _queue.pop();
    if (_queue.empty())
      return std::chrono::microseconds(-1);

    entry = _jobs.find(_next_run->job_id());
  }

  _next_run = _queue.top();
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
      _next_run->time() - now);
}

void FunctionScheduler::run() {
  std::unique_lock<std::mutex> lock(_mutex);

  while (_running) {
    if (_queue.empty()) {
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
      // before the current ´_next_run´
      if (_cond.wait_for(lock, wait_time) == std::cv_status::timeout) {
        // Lock timed out, i.e., no new run was added while we waited.
        // The run selected as next is still the correct one.
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
      _next_run == _queue.top(), "Next run does not match queue top.");
  _queue.pop(); // Remove this run from the queue

  // Check if the job was canceled in the meantime.
  auto entry = _jobs.find(_next_run->job_id());
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

int FunctionScheduler::id() {
  return _current_id++;
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
  auto run = std::make_shared<Run>(job_id, time);
  _queue.push(std::move(run));

  // Notify the thread handling run execution.
  if (_running)
    _cond.notify_one();
}

int FunctionScheduler::scheduleJob(std::unique_ptr<Job> job) {
  std::unique_lock<std::mutex> lock(_mutex);
  int job_id = id();

  if (_running) {
    addRun(lock, job_id, job);
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
  auto now = std::chrono::steady_clock::now();
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
  while (!_queue.empty())
    _queue.pop();

  // reset counters
  for (const auto& entry : _jobs) {
    entry.second->reset_counter();
  }
  return 1;
}

int FunctionScheduler::pause() {
  if (_paused || !_running)
    return -1;

  _running = false;
  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // exits the loop.
  _cond.notify_one();
  if (_thread.joinable()) {
    _thread.join();
  }

  _paused_time = std::chrono::steady_clock::now();
  _paused = true;
  return 1;
}

int FunctionScheduler::resume() {
  if (!_paused)
    return -1;

  auto diff = std::chrono::steady_clock::now() - _paused_time;
  auto _queue_copy = _queue;
  while (!_queue_copy.empty()) {
    auto entry = _queue_copy.top();
    _queue_copy.pop();
    entry->set_time(entry->time() + diff);
  }

  _running = true;
  _paused = false;
  _thread = std::thread(&FunctionScheduler::run, this);
  return 1;
}

bool FunctionScheduler::isRunning() const {
  return _running;
}

int FunctionScheduler::currentId() const {
  return _current_id;
}

} // namespace c10
