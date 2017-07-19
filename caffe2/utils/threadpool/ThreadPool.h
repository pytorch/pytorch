#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include "ThreadPoolCommon.h"

#ifndef CAFFE2_THREADPOOL_MOBILE
#error "mobile build state not defined"
#endif

// ThreadPool only used in mobile builds at the moment
#if CAFFE2_THREADPOOL_MOBILE

// Compile-time flag to control inclusion of per-worker thread stats
// #define CAFFE2_THREADPOOL_STATS

// Compile-time flag to control usage of main thread work imbalance
// #define CAFFE2_THREADPOOL_MAIN_IMBALANCE

#include <stdlib.h> // posix_memalign
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2 {

constexpr size_t kCacheLineSize = 64;

template <typename T>
struct AllocAligned {
  // Allocate a T aligned at an `align` byte address
  template <typename... Args>
  static T* alloc(size_t align, Args&&... args) {
    void* p = nullptr;
// FIXME: we should just be able to use std::align
#if !defined(__ANDROID__)
    posix_memalign((void**)&p, align, sizeof(T));
#else
    p = memalign(align, sizeof(T));
#endif

    if (p) {
      return new (p) T(std::forward<Args>(args)...);
    }

    return nullptr;
  }

  // Free a T previously allocated via AllocAligned<T>::alloc()
  static void release(T* p) {
    if (p) {
      p->~T();
      free((void*)p);
    }
  }
};

// Deleter object for unique_ptr for an aligned object
template <typename T>
struct AlignedDeleter {
  void operator()(T* p) const {
    AllocAligned<T>::release(p);
  }
};

// make_unique that guarantees alignment
template <typename T>
struct MakeAligned {
  template <typename... Args>
  static std::unique_ptr<T, AlignedDeleter<T>> make(
      size_t align,
      Args&&... args) {
    return std::unique_ptr<T, AlignedDeleter<T>>(
        AllocAligned<T>::alloc(align, std::forward<Args>(args)...));
  }
};

struct ThreadPool;

#ifdef CAFFE2_THREADPOOL_STATS
struct ThreadStats {
  inline ThreadStats() : numAssigned(0), numWorkedOn(0), numStolen(0) {}

  inline void reset() {
    numAssigned = 0;
    numWorkedOn = 0;
    numStolen = 0;
  }

  int numAssigned;
  int numWorkedOn;
  int numStolen;
};
#endif

struct alignas(kCacheLineSize) ThreadInfo {
  inline ThreadInfo(int threadId, int numThreads)
      : rangeStart_(0),
        rangeEnd_(0),
        rangeLength_(0),
        wantExit_(false),
        threadId_(threadId),
        numThreads_(numThreads) {}

  // Entry point for all worker threads
  void threadMain(int threadId, ThreadPool* pool);

  // Runs a task, and when we're done with our local queue, steal from
  // neighbors.
  // Returns true if all work is done (we were the last thread to do
  // work)
  bool runAndSteal(int threadId, ThreadPool* pool);

  // Index of first element in the work range.
  // Before processing a new element the owning worker thread
  // increments this value.
  long rangeStart_;

  // Index of the element after the last element of the work range.
  // Before processing a new element the stealing worker thread
  // decrements this value.
  std::atomic<long> rangeEnd_;

  // The number of elements in the work range.
  // Due to race conditions range_length <= range_end - range_start.
  // The owning worker thread must decrement this value before
  // incrementing @a range_start.
  // The stealing worker thread must decrement this value before
  // decrementing @a range_end.
  std::atomic<long> rangeLength_;

  // Should this thread exit?
  bool wantExit_;

  // Our thread index
  int threadId_;

  // How many threads are there in total?
  int numThreads_;

#ifdef CAFFE2_THREADPOOL_STATS
  // Updated stats
  ThreadStats stats_;
#endif
};

class alignas(kCacheLineSize) ThreadPool {
 public:
  // Constructs a work-stealing threadpool with the given number of
  // threads
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  ThreadPool(int numThreads);

  // Shuts down all worker threads (if any) before destroying ourselves
  ~ThreadPool();

  // Returns the number of threads currently in use
  int getNumThreads() const;

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
  void setMinWorkSize(size_t size);
  size_t getMinWorkSize() const {
    return minWorkSize_;
  }

#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
  // Set imbalance factor for the main thread versus other threads;
  // default is 1.25
  void setImbalanceRatio(float ratio);
#endif

  // Called to schedule work on the threadpool
  void run(const std::function<void(int, size_t)>& fn, size_t range);

#ifdef CAFFE2_THREADPOOL_STATS
  // Returns current per-thread statistics. If reset is true, reset
  // current values.
  std::vector<ThreadStats> getStats(bool reset = false);
#endif

 protected:
  friend struct ThreadInfo;

  // What we are currently working on
  const std::function<void(int, size_t)>* fn_;

  // How many work items are outstanding? When this reaches 0, our
  // main thread is resumed
  std::atomic<long> workItemsPending_;

  // Current work ID that we're running; sequentially increments
  long currentWorkId_;

  // Mutex that guards all monitors and state updates
  std::mutex mutex_;

  // Main thread waits on this before running new work, to make sure
  // that all worker threads have looped back around to await new work
  std::condition_variable threadReadyMonitor_;

  // All worker threads wait on this to make sure that they have work
  // available for processing
  std::condition_variable threadStartMonitor_;

  // Main thread waits on this before returning to the thread pool
  // caller; note that we don't actually wait on the worker threads
  // saying that they're all done (woken up); we only check when the
  // thread pool is called again
  std::condition_variable threadDoneMonitor_;

  // How many threads are ready to process new work?
  size_t threadsReady_;

  // The first entry is always for the main thread
  std::vector<std::unique_ptr<ThreadInfo, AlignedDeleter<ThreadInfo>>>
      threadInfo_;

  // Set of threads that we are managing
  std::vector<std::thread> threads_;

  // What's the minimum work size for using the threadpool?
  size_t minWorkSize_;

#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
  // Imbalance factor for main vs. other thread work
  float imbalanceRatio_;
#endif

  // Mutex that ensures that only one user call to the ThreadPool is
  // outstanding
  mutable std::mutex executionMutex_;
};

} // namespace caffe2

#endif // CAFFE2_THREADPOOL_MOBILE

#endif // CAFFE2_UTILS_THREADPOOL_H_
