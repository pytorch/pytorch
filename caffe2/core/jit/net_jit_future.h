#ifndef CAFFE2_NET_JIT_FUTURE_H
#define CAFFE2_NET_JIT_FUTURE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>

namespace caffe2 {

class JITFuture {
 public:
  JITFuture();
  // Creates a future completed when all given futures are completed
  explicit JITFuture(const std::vector<JITFuture*>& futures);

  JITFuture(const JITFuture&) = delete;
  JITFuture& operator=(const JITFuture&) = delete;

  bool IsCompleted() const;

  bool IsFailed() const;

  std::string ErrorMessage() const;

  void Wait() const;

  void SetCallback(std::function<void(const JITFuture*)> callback);

  void SetCompleted(const char* err_msg = nullptr);

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_completed_;
  std::atomic<bool> completed_;
  std::atomic<bool> failed_;
  std::string err_msg_;
  std::vector<std::function<void(const JITFuture*)>> callbacks_;

  struct ParentCounter {
    explicit ParentCounter(int init_parent_count)
        : parent_count(init_parent_count), parent_failed(false) {}
    std::atomic<int> parent_count;
    std::mutex err_mutex;
    std::atomic<bool> parent_failed;
    std::string err_msg;
  };
};

} // namespace caffe2

#endif // CAFFE2_NET_JIT_FUTURE_H
