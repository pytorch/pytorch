#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include "ThreadPoolCommon.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "caffe2/core/common.h"

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2 {

struct Task;
class WorkersPool;

constexpr size_t kCacheLineSize = 64;

// A threadpool with the given number of threads.
// NOTE: the kCacheLineSize alignment is present only for cache
// performance, and is not strictly enforced (for example, when
// the object is created on the heap). Thus, in order to avoid
// misaligned intrinsics, no SSE instructions shall be involved in
// the ThreadPool implementation.
// Note: alignas is disabled because some compilers do not deal with
// TORCH_API and alignas annotations at the same time.
class TORCH_API /*alignas(kCacheLineSize)*/ ThreadPool {
 public:
  static ThreadPool* createThreadPool(int numThreads);
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  virtual ~ThreadPool() = default;
  // Returns the number of threads currently in use
  virtual int getNumThreads() const = 0;
  virtual void setNumThreads(size_t numThreads) = 0;

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
  void setMinWorkSize(size_t size) {
    std::lock_guard<std::mutex> guard(executionMutex_);
    minWorkSize_ = size;
  }

  size_t getMinWorkSize() const {
    return minWorkSize_;
  }
  virtual void run(const std::function<void(int, size_t)>& fn, size_t range) = 0;

  // Run an arbitrary function in a thread-safe manner accessing the Workers
  // Pool
  virtual void withPool(const std::function<void(WorkersPool*)>& fn) = 0;

 protected:
  static size_t defaultNumThreads_;
  mutable std::mutex executionMutex_;
  size_t minWorkSize_;
};

} // namespace caffe2

#endif // CAFFE2_UTILS_THREADPOOL_H_
