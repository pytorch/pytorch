#include "caffe2/utils/threadpool/ThreadPool.h"
#include "caffe2/core/logging.h"

#if CAFFE2_THREADPOOL_MOBILE

namespace caffe2 {

// Default smallest amount of work that will be partitioned between
// multiple threads; the runtime value is configurable
constexpr size_t kDefaultMinWorkSize = 80;

ThreadPool::ThreadPool(int numThreads)
    : fn_(nullptr),
      workItemsPending_(0),
      currentWorkId_(0),
      threadsReady_(0),
      minWorkSize_(kDefaultMinWorkSize) {
  std::lock_guard<std::mutex> guard(mutex_);

  // All worker threads (and the main thread) have a ThreadInfo
  for (auto i = 0; i < numThreads; ++i) {
    threadInfo_.emplace_back(
      std::unique_ptr<ThreadInfo>(new ThreadInfo(i, numThreads)));
  }

  // The first ThreadInfo is for the main thread
  for (auto i = 1; i < numThreads; ++i) {
    auto pInfo = &(threadInfo_[i]);
    auto fn = [pInfo, this, i]() {
      (*pInfo)->threadMain(i, this);
    };

    threads_.emplace_back(std::thread(std::move(fn)));
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& info : threadInfo_) {
      info->wantExit_ = true;
    }
  }

  threadStartMonitor_.notify_all();

  // Wait on all threads to exit
  for (auto& thread : threads_) {
    thread.join();
  }
}

int
ThreadPool::getNumThreads() const {
  std::lock_guard<std::mutex> guard(executionMutex_);

  return threadInfo_.size();
}

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
void
ThreadPool::setMinWorkSize(size_t size) {
  std::lock_guard<std::mutex> guard(executionMutex_);

  minWorkSize_ = size;
}

void
ThreadPool::run(const std::function<void(int, size_t)>& fn, size_t range) {
  std::lock_guard<std::mutex> guard(executionMutex_);

  // If there are no worker threads, or if the range is too small (too
  // little work), just run locally
  if (threads_.size() == 0 || range < minWorkSize_) {
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }

    return;
  }

  // Set up thread state
  {
    std::unique_lock<std::mutex> lock(mutex_);

    // We've guaranteed that all threads have finished work for the
    // previous round, but we don't want threads to read new work
    // information out of order. Wait for all of the old threads to
    // check in first
    while (threadsReady_ < threads_.size()) {
      threadReadyMonitor_.wait(lock);
    }

    // Our threads are ready, and are waiting for us to start them.
    threadsReady_ = 0;

    fn_ = &fn;

    auto numThreads = threadInfo_.size();
    size_t workUnitsPerThread = (numThreads + range - 1) / numThreads;

    for (size_t i = 0; i < numThreads; ++i) {
      auto& threadInfo = threadInfo_[i];

      threadInfo->rangeStart_ = std::min(i * workUnitsPerThread, range);
      threadInfo->rangeEnd_ = std::min((i + 1) * workUnitsPerThread, range);
      threadInfo->rangeLength_ =
        threadInfo->rangeEnd_ - threadInfo->rangeStart_;
    }

    workItemsPending_ = range;
    ++currentWorkId_;
  }

  // Wake all worker threads
  threadStartMonitor_.notify_all();

  // We participate as well
  bool done = threadInfo_[0]->runAndSteal(0, this);

  // This thread may have been the one to finish all the work
  if (!done) {
    // Wait until we get signalled back
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (workItemsPending_.load() > 0) {
        threadDoneMonitor_.wait(lock);
      }
    }
  }
}

void
ThreadInfo::threadMain(int threadId, ThreadPool* pool) {
  long lastProcessedWorkId = 0;

  while (true) {
    {
      // Kick off
      std::unique_lock<std::mutex> lock(pool->mutex_);
      int numAtBarrier = ++(pool->threadsReady_);
      // numThreads includes main thread, we only care about the # of
      // worker threads here
      if (numAtBarrier == (numThreads_ - 1)) {
        pool->threadReadyMonitor_.notify_one();
      }

      // Wait on main to give us new work
      while (!wantExit_ && pool->currentWorkId_ <= lastProcessedWorkId) {
        pool->threadStartMonitor_.wait(lock);
      }

      // Whether or not we actually do some work, this is the new work
      // item we're handling
      lastProcessedWorkId = pool->currentWorkId_;
    }

    if (wantExit_) {
      return;
    }

    bool shouldSignal = runAndSteal(threadId, pool);

    if (shouldSignal) {
      std::lock_guard<std::mutex> guard(pool->mutex_);
      pool->threadDoneMonitor_.notify_one();
    }
  }
}

bool
ThreadInfo::runAndSteal(int threadId, ThreadPool* pool) {
  auto lambdaFunctionToRun = pool->fn_;
  auto localItemsCompleted = 0;

  /* Process thread's own range of items */
  auto curItem = rangeStart_;
  while (true) {
    auto curRangeLength = --rangeLength_; // atomic

    if (curRangeLength < 0) {
      // someone stole all of our work
      break;
    }

    (*lambdaFunctionToRun)(threadId, curItem);

    ++curItem;
    ++localItemsCompleted;
  }

  /* Done, now look for other threads' items to steal */
  for (auto i = (threadId_ + 1) % numThreads_;
       i != threadId_;
       i = (i + 1) % numThreads_) {
    auto& otherThread = pool->threadInfo_[i];

    while (true) {
      auto curRangeLength = --(otherThread->rangeLength_); // atomic

      if (curRangeLength < 0) {
        break;
      }

      // We're successfully stealing a work item from the other thread
      auto itemId = --(otherThread->rangeEnd_); // atomic

      (*lambdaFunctionToRun)(threadId, itemId);
      ++localItemsCompleted;
    }
  }

  if (localItemsCompleted > 0) {
    auto numRemaining =
      (pool->workItemsPending_ -= localItemsCompleted); // atomic
    DCHECK_GE(numRemaining, 0);

    if (numRemaining == 0) {
      // We were the last thread to finish all work
      return true;
    }
  }

  return false;
}

} // namespace caffe2

#endif // CAFFE2_THREADPOOL_MOBILE
