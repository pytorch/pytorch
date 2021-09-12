#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/numa.h>
#include <c10/util/thread_name.h>

namespace c10 {

// TODO: move this to C10 and make it C10_API
class C10_API TaskThreadPoolBase {
 public:
  virtual void run(std::function<void()> func) = 0;

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

  static size_t defaultNumThreads() {
    auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    num_threads /= 2;
#endif
    return num_threads;
  }
};

class C10_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
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
      int pool_size,
      int numa_node_id = -1,
      std::function<void()> init_thread = nullptr);

  ~ThreadPool();

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(std::function<void()> func) override;

  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

 private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};

class C10_API TaskThreadPool : public c10::ThreadPool {
 public:
  explicit TaskThreadPool(std::size_t pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          setThreadName("CaffeTaskThread");
          NUMABind(numa_node_id);
        }) {}
};

C10_DECLARE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);

} // namespace c10
