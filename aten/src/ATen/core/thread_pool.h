#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {

namespace ivalue {
struct Future;
} // namespace ivalue

// TODO: move this to C10 and make it C10_API
class CAFFE2_API TaskThreadPoolBase {
 public:
  virtual void run(const std::function<void()>& func) = 0;

  virtual size_t size() const = 0;

  /**
   * The number of available (i.e. idle) threads in this thread pool.
   */
  virtual size_t numAvailable() const = 0;

  /**
   * Check if the current thread is from the thread pool.
   */
  virtual bool inThreadPool() const = 0;

  virtual ~TaskThreadPoolBase() noexcept {}
};

class CAFFE2_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
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
  std::atomic_bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

 public:
  ThreadPool() = delete;

  explicit ThreadPool(
      std::size_t pool_size,
      int numa_node_id = -1);

  ~ThreadPool();

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(const std::function<void()>& func) override;

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
  void waitWorkComplete();

 protected:
  virtual void init_thread() {}

 private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};

CAFFE2_API void setNumThreads(size_t v);

CAFFE2_API ThreadPool& global_work_queue();

} // namespace c10
