#include <c10/core/thread_pool.h>

namespace c10 {

ThreadPool::ThreadPool(std::size_t pool_size, int numa_node_id)
    : threads_(pool_size),
      running_(true),
      complete_(true),
      available_(threads_.size()),
      total_(threads_.size()),
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

// constexpr initialization guaranteed to be before any static initialization
std::atomic<int> num_threads{1};
void setNumThreads(size_t v) {
  if(-1  == num_threads.exchange(v)) {
   throw std::runtime_error("Error: cannot set num threads after pool has started");
  }
}

TaskThreadPoolBase& global_work_queue() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create("C10", 0, num_threads.exchange(-1), false);
  return *pool;
}

C10_DEFINE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);

namespace {

std::shared_ptr<TaskThreadPoolBase> createC10ThreadPool(
    int device_id,
    int pool_size,
    bool create_new) {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      std::make_shared<ThreadPool>(pool_size);
  return pool;
}

} // namespace

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, createC10ThreadPool);

} // namespace c10
