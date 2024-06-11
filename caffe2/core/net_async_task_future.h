#ifndef CAFFE2_NET_ASYNC_TASK_FUTURE_H
#define CAFFE2_NET_ASYNC_TASK_FUTURE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caffe2 {

// Represents the state of AsyncTask execution, that can be queried with
// IsCompleted/IsFailed. Callbacks are supported through SetCallback and
// are called upon future's completion.

class AsyncTaskFuture {
 public:
  AsyncTaskFuture();
  // Creates a future completed when all given futures are completed
  explicit AsyncTaskFuture(const std::vector<AsyncTaskFuture*>& futures);
  ~AsyncTaskFuture();

  AsyncTaskFuture(const AsyncTaskFuture&) = delete;

  AsyncTaskFuture& operator=(const AsyncTaskFuture&) = delete;

  bool IsCompleted() const;

  bool IsFailed() const;

  std::string ErrorMessage() const;

  void Wait() const;

  void SetCallback(std::function<void(const AsyncTaskFuture*)> callback);

  void SetCompleted(const char* err_msg = nullptr);

  void ResetState();

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_completed_;
  std::atomic<bool> completed_;
  std::atomic<bool> failed_;
  std::string err_msg_;
  std::vector<std::function<void(const AsyncTaskFuture*)>> callbacks_;

  struct ParentCounter {
    explicit ParentCounter(int init_parent_count)
        : init_parent_count_(init_parent_count),
          parent_count(init_parent_count),
          parent_failed(false) {}

    void Reset() {
      std::unique_lock<std::mutex> lock(err_mutex);
      parent_count = init_parent_count_;
      parent_failed = false;
      err_msg = "";
    }

    const int init_parent_count_;
    std::atomic<int> parent_count;
    std::mutex err_mutex;
    std::atomic<bool> parent_failed;
    std::string err_msg;
  };

  std::unique_ptr<ParentCounter> parent_counter_;
};

} // namespace caffe2

#endif // CAFFE2_NET_ASYNC_TASK_FUTURE_H
