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
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  ThreadPool(int numThreads);
  ~ThreadPool();
  // Returns the number of threads currently in use
  int getNumThreads() const;
  void setNumThreads(size_t numThreads);

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
  void setMinWorkSize(size_t size);
  size_t getMinWorkSize() const {
    return minWorkSize_;
  }
  void run(const std::function<void(int, size_t)>& fn, size_t range);

  // Run an arbitrary function in a thread-safe manner accessing the Workers
  // Pool
  void withPool(const std::function<void(WorkersPool*)>& fn);

 private:
  static size_t defaultNumThreads_;
  mutable std::mutex executionMutex_;
  size_t minWorkSize_;
  std::atomic_size_t numThreads_;
  std::shared_ptr<WorkersPool> workersPool_;
  std::vector<std::shared_ptr<Task>> tasks_;
};

} // namespace caffe2

#endif // CAFFE2_UTILS_THREADPOOL_H_
