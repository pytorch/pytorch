#pragma once
#include <ATen/Parallel.h>
#include <c10/core/thread_pool.h>

#if USE_EIGEN_THREADPOOL
#include "unsupported/Eigen/CXX11/ThreadPool"
#else
#endif

namespace at {

#if USE_EIGEN_THREADPOOL

struct PTThreadPoolEnvironment {
  struct Task {
    std::function<void()> f;
  };

  using EnvThread = std::thread;

  EnvThread* CreateThread(std::function<void()> func) {
    return new std::thread([func]() {
      c10::setThreadName("PTThreadPool-Eigen");
      at::init_num_threads();
      func();
    });
  }

  Task CreateTask(std::function<void()> func) {
    return Task { func };
  }

  void ExecuteTask(Task task) {
    task.f();
  }
};

struct CAFFE2_API PTThreadPool
    : Eigen::ThreadPoolTempl<PTThreadPoolEnvironment>, TaskThreadPoolBase {

  explicit PTThreadPool(
      int pool_size,
      int numa_node_id = -1) :
    Eigen::ThreadPoolTempl<PTThreadPoolEnvironment>(
        pool_size < 0 ? defaultNumThreads() : pool_size, false) {}
  // TODO: extra ctor params

  void run(const std::function<void()>& func) override {
    Schedule(func);
  }

  size_t size() const override {
    return NumThreads();
  }

  size_t numAvailable() const override {
    // treating all threads as available
    return NumThreads();
  }

  bool inThreadPool() const override {
    return CurrentThreadId() != -1;
  }
};

#else

class CAFFE2_API PTThreadPool : public c10::ThreadPool {
 public:
  explicit PTThreadPool(
      int pool_size,
      int numa_node_id = -1)
    : c10::ThreadPool(pool_size, numa_node_id) {}

  void init_thread() override {
    c10::setThreadName("PTThreadPool");
    at::init_num_threads();
  }
};

#endif

} // namespace at
