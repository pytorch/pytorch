#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <c10/macros/Export.h>
#include <c10/util/Registry.h>
#include <c10/util/numa.h>
#include <c10/util/thread_name.h>

namespace c10 {

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

  virtual ~TaskThreadPoolBase() noexcept = default;

  static size_t defaultNumThreads();
};

class C10_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  std::queue<std::function<void()>> tasks_;
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
      const std::function<void()>& init_thread = nullptr);

  ~ThreadPool() override;

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(std::function<void()> func) override;

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

 private:
  // @brief Entry point for pool threads.
  void main_loop();
};

class C10_API TaskThreadPool : public c10::ThreadPool {
 public:
  explicit TaskThreadPool(int pool_size, int numa_node_id = -1)
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
