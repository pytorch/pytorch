#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include "ThreadPoolCommon.h"

#ifndef CAFFE2_THREADPOOL_MOBILE
#error "mobile build state not defined"
#endif

// ThreadPool only used in mobile builds at the moment
#if CAFFE2_THREADPOOL_MOBILE

#include <memory>
#include <mutex>
#include <vector>

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2 {

class Task;
class WorkersPool;

constexpr size_t kCacheLineSize = 64;

class alignas(kCacheLineSize) ThreadPool {
 public:
  // Constructs a work-stealing threadpool with the given number of
  // threads
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  ThreadPool(int numThreads);
  ~ThreadPool();
  // Returns the number of threads currently in use
  int getNumThreads() const;

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
  void setMinWorkSize(size_t size);
  size_t getMinWorkSize() const { return minWorkSize_; }
  void run(const std::function<void(int, size_t)>& fn, size_t range);

private:
  mutable std::mutex executionMutex_;
  size_t minWorkSize_;
  size_t numThreads_;
  std::shared_ptr<WorkersPool> workersPool_;
  std::vector<std::shared_ptr<Task>> tasks_;
};

} // namespace caffe2

#endif // CAFFE2_THREADPOOL_MOBILE

#endif // CAFFE2_UTILS_THREADPOOL_H_
