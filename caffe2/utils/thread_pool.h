#ifndef CAFFE2_UTILS_THREAD_POOL_H_
#define CAFFE2_UTILS_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include "caffe2/core/numa.h"

namespace caffe2 {

class TaskThreadPool {
 private:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(const std::function<void()>& f)
        : run_with_id(false), no_id(f), with_id(nullptr) {}
    explicit task_element_t(const std::function<void(std::size_t)>& f)
        : run_with_id(true), no_id(nullptr), with_id(f) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

 public:
  explicit TaskThreadPool(std::size_t pool_size, int numa_node_id = -1)
      : threads_(pool_size),
        running_(true),
        complete_(true),
        available_(pool_size),
        total_(pool_size),
        numa_node_id_(numa_node_id) {
    for (std::size_t i = 0; i < pool_size; ++i) {
      threads_[i] = std::thread(std::bind(&TaskThreadPool::main_loop, this, i));
    }
  }

  // Set running flag to false then notify all threads.
  ~TaskThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      running_ = false;
      condition_.notify_all();
    }

    try {
      for (auto& t : threads_) {
        t.join();
      }
    } catch (const std::exception&) {
    }
  }

  /// @brief Add task to the thread pool if a thread is currently available.
  template <typename Task>
  void runTask(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.push(task_element_t(static_cast<std::function<void()>>(task)));
    complete_ = false;
    condition_.notify_one();
  }

  void run(const std::function<void()>& func) {
    runTask(func);
  }

  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.push(
        task_element_t(static_cast<std::function<void(std::size_t)>>(task)));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!complete_) {
      completed_.wait(lock);
    }
  }

 private:
  /// @brief Entry point for pool threads.
  void main_loop(std::size_t index) {
    NUMABind(numa_node_id_);

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
};

} // namespace caffe2

#endif // CAFFE2_UTILS_THREAD_POOL_H_
