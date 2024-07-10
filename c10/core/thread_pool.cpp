#include <c10/core/thread_pool.h>
#include <c10/util/Logging.h>
#if !defined(__powerpc__) && !defined(__s390x__)
#include <cpuinfo.h>
#endif

namespace c10 {

size_t TaskThreadPoolBase::defaultNumThreads() {
  size_t num_threads = 0;
#if !defined(__powerpc__) && !defined(__s390x__)
  if (cpuinfo_initialize()) {
    // In cpuinfo parlance cores are physical ones and processors are virtual
    // ThreadPool should be defaulted to number of physical cores
    size_t num_cores = cpuinfo_get_cores_count();
    num_threads = cpuinfo_get_processors_count();
    if (num_cores > 0 && num_cores < num_threads) {
      return num_cores;
    }
    if (num_threads > 0) {
      return num_threads;
    }
  }
#endif
  num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  return num_threads;
}

ThreadPool::ThreadPool(
    int pool_size,
    int numa_node_id,
    const std::function<void()>& init_thread)
    : threads_(pool_size < 0 ? defaultNumThreads() : pool_size),
      running_(true),
      complete_(true),
      available_(threads_.size()),
      total_(threads_.size()),
      numa_node_id_(numa_node_id) {
  for (std::size_t i = 0; i < threads_.size(); ++i) {
    threads_[i] = std::thread([this, i, init_thread]() {
      if (init_thread) {
        init_thread();
      }
      this->main_loop(i);
    });
  }
}

ThreadPool::~ThreadPool() {
  // Set running flag to false then notify all threads.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
  }

  for (auto& t : threads_) {
    try {
      t.join();
    } catch (const std::exception&) {
    }
  }
}

size_t ThreadPool::size() const {
  return threads_.size();
}

size_t ThreadPool::numAvailable() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return available_;
}

bool ThreadPool::inThreadPool() const {
  for (auto& thread : threads_) {
    if (thread.get_id() == std::this_thread::get_id()) {
      return true;
    }
  }
  return false;
}

void ThreadPool::run(std::function<void()> func) {
  if (threads_.empty()) {
    throw std::runtime_error("No threads to run a task");
  }
  std::unique_lock<std::mutex> lock(mutex_);

  // Set task and signal condition variable so that a worker thread will
  // wake up and use the task.
  tasks_.emplace(std::move(func));
  complete_ = false;
  condition_.notify_one();
}

void ThreadPool::waitWorkComplete() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [&]() { return complete_; });
}

void ThreadPool::main_loop(std::size_t index) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (running_) {
    // Wait on condition variable while the task is empty and
    // the pool is still running.
    condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
    // If pool is no longer running, break out of loop.
    if (!running_) {
      break;
    }

    // Copy task locally and remove from the queue.  This is
    // done within its own scope so that the task object is
    // destructed immediately after running the task.  This is
    // useful in the event that the function contains
    // shared_ptr arguments bound via bind.
    {
      task_element_t tasks = std::move(tasks_.front());
      tasks_.pop();
      // Decrement count, indicating thread is no longer available.
      --available_;

      lock.unlock();

      // Run the task.
      try {
        if (tasks.run_with_id) {
          tasks.with_id(index);
        } else {
          tasks.no_id();
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Exception in thread pool task: " << e.what();
      } catch (...) {
        LOG(ERROR) << "Exception in thread pool task: unknown";
      }

      // Destruct tasks before taking the lock.  As tasks
      // are user provided std::function, they can run
      // arbitrary code during destruction, including code
      // that can reentrantly call into ThreadPool (which would
      // cause a deadlock if we were holding the lock).
    }

    // Update status of empty, maybe
    // Need to recover the lock first
    lock.lock();

    // Increment count, indicating thread is available.
    ++available_;
    if (tasks_.empty() && available_ == total_) {
      complete_ = true;
      completed_.notify_one();
    }

    // Deliberately hold the lock on the backedge, so this thread has an
    // opportunity to acquire a new task before another thread acquires
    // the lock.
  } // while running_
}

C10_DEFINE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);
} // namespace c10
