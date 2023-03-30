#include <torch/csrc/lazy/core/thread_pool.h>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>

namespace torch {
namespace lazy {
namespace {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    threads_.reserve(num_threads);
    for (const auto i : c10::irange(num_threads)) {
      (void)i; // Suppress unused variable warning
      threads_.emplace_back([this]() { Worker(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exiting_ = true;
      cv_.notify_all();
    }
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void Schedule(std::function<void()> closure) {
    // If we have more work scheduled than waiting worker threads, just schedule
    // it on a separate thread. This prevents tricky thread-pool-size-deadlocks
    // caused by an undersized thread pool and closures that end up doing sync
    // waits on the pool threads.
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (work_.size() < waiting_) {
        work_.emplace_back(std::move(closure));
        lock.unlock();
        cv_.notify_one();
        return;
      }
    }
    ScheduleOnThread(std::move(closure));
  }

 private:
  void Worker() {
    while (true) {
      std::function<void()> closure = GetWork();
      if (closure == nullptr) {
        break;
      }
      try {
        closure();
      } catch (const std::exception& ex) {
        TORCH_LAZY_COUNTER("ThreadPoolException", 1);
        LOG(ERROR) << "Exception from running thread pool closure: "
                   << ex.what();
      }
    }
  }

  void ScheduleOnThread(std::function<void()> closure) {
    std::thread thread(std::move(closure));
    thread.detach();
  }

  std::function<void()> GetWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_;
    cv_.wait(lock, [this] { return exiting_ || !work_.empty(); });
    --waiting_;
    if (work_.empty()) {
      return nullptr;
    }
    std::function<void()> closure(std::move(work_.front()));
    work_.pop_front();
    return closure;
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool exiting_ = false;
  std::deque<std::function<void()>> work_;
  size_t waiting_ = 0;
};

ThreadPool* GetIoThreadPool() {
  static ThreadPool* pool =
      new ThreadPool(FLAGS_torch_lazy_io_thread_pool_size);
  return pool;
}

} // namespace

class Completion::Data {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_; });
    if (exptr_ != nullptr) {
      std::rethrow_exception(exptr_);
    }
  }

  static std::function<void()> GetCompleter(
      std::shared_ptr<Data> data,
      std::function<void()> closure) {
    auto closure_wrapper = [closure = std::move(closure), data]() {
      std::exception_ptr exptr;
      try {
        closure();
      } catch (...) {
        exptr = std::current_exception();
      }
      data->Complete(exptr);
    };
    return closure_wrapper;
  }

 private:
  void Complete(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::move(exptr);
    completed_ = true;
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exptr_;
};

Completion::Completion(std::shared_ptr<Data> data) : data_(std::move(data)) {}

void Completion::Wait() {
  data_->Wait();
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}

Completion ScheduleIoClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetIoThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

} // namespace lazy
} // namespace torch
