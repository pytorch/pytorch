#include <ATen/core/ivalue.h>
#include <ATen/core/thread_pool.h>

namespace c10 {

ThreadPool::ThreadPool(c10::optional<std::size_t> pool_size, int numa_node_id)
    : threads_(pool_size.has_value() ? pool_size.value() : 1),
      running_(true),
      complete_(true),
      available_(pool_size.has_value() ? pool_size.value() : 1),
      total_(pool_size.has_value() ? pool_size.value() : 1),
      numa_node_id_(numa_node_id) {
  for (std::size_t i = 0; i < threads_.size(); ++i) {
    threads_[i] = std::thread(std::bind(&ThreadPool::main_loop, this, i));
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

void ThreadPool::run(const std::function<void()>& func) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Set task and signal condition variable so that a worker thread will
  // wake up and use the task.
  tasks_.push(task_element_t(func));
  complete_ = false;
  condition_.notify_one();
}

void ThreadPool::waitWorkComplete() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!complete_) {
    completed_.wait(lock);
  }
}

void ThreadPool::main_loop(std::size_t index) {
  init_thread();

  while (running_) {
    // Wait on condition variable while the task is empty and
    // the pool is still running.
    std::unique_lock<std::mutex> lock(mutex_);
    while (tasks_.empty() && running_) {
      condition_.wait(lock);
    }
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
      auto tasks = tasks_.front();
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
      } catch (const std::exception&) {
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
    }
  } // while running_
}

c10::optional<size_t> num_threads(c10::optional<size_t> val) {
  static c10::optional<size_t> n_threads = val;
  if (val.has_value() && val != n_threads) {
    throw std::runtime_error(
        "Attempted to set number of ATen threads"
        " multiple times.");
  }
  return n_threads;
}

ThreadPool& global_work_queue() {
  static ThreadPool thread_pool(num_threads());

  return thread_pool;
}

} // namespace c10
