#include "caffe2/utils/threadpool/ThreadPool.h"
#include "caffe2/core/logging.h"

#if CAFFE2_ANDROID
#include <cpu-features.h>
#endif

CAFFE2_DEFINE_bool(
    caffe2_threadpool_force_inline,
    false,
    "Force to always run jobs on the calling thread");

// Whether or not threadpool caps apply to Android
CAFFE2_DEFINE_int(caffe2_threadpool_android_cap, true, "");

// Whether or not threadpool caps apply to iOS
CAFFE2_DEFINE_int(caffe2_threadpool_ios_cap, false, "");

#if CAFFE2_THREADPOOL_MOBILE

namespace caffe2 {

// Default smallest amount of work that will be partitioned between
// multiple threads; the runtime value is configurable
#if CAFFE2_ANDROID
constexpr size_t kDefaultMinWorkSize = 8;
#else
constexpr size_t kDefaultMinWorkSize = 80;
#endif

#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
constexpr float kDefaultImbalanceRatio = 1.0f;
#endif

std::unique_ptr<ThreadPool> ThreadPool::defaultThreadPool() {
  int numThreads = std::thread::hardware_concurrency();

#ifdef CAFFE2_ANDROID
  // std::thread::hardware_concurrency returns online cores
  // (sysconf(_SC_NPROCESSORS_ONLN)), but we want the total number of CPUs. In
  // most cases they will match, but since the threadpool is instantiated once,
  // we want the number of threads for each device to be predictable.
  int numCpus = android_getCpuCount();
  LOG(INFO) << "Android cpu count: " << numCpus
            << ", hardware_concurrency: " << numThreads;
  numThreads = numCpus;
#endif

  bool applyCap = false;
#if CAFFE2_ANDROID
  applyCap = caffe2::FLAGS_caffe2_threadpool_android_cap;
#elif CAFFE2_IOS
  applyCap = caffe2::FLAGS_caffe2_threadpool_ios_cap;
#else
#error Undefined architecture
#endif

  if (applyCap) {
    // 1 core  -> 1 thread
    // 2 cores -> 2 threads
    // 4 cores -> 2 threads
    // 8 cores -> 4 threads
    // more, continue limiting to half of available cores

    if (numThreads <= 3) {
      // no change
    } else if (numThreads <= 5) {
      // limit to 2
      numThreads = 2;
    } else {
      // Use half the cores
      numThreads = numThreads / 2;
    }
  }
  LOG(INFO) << "Constructing thread pool with " << numThreads << " threads";
  return caffe2::make_unique<ThreadPool>(numThreads);
}

ThreadPool::ThreadPool(int numThreads)
    : fn_(nullptr),
      workItemsPending_(0),
      currentWorkId_(0),
      threadsReady_(0),
      minWorkSize_(kDefaultMinWorkSize)
#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
      ,
      imbalanceRatio_(kDefaultImbalanceRatio)
#endif
{
  std::lock_guard<std::mutex> guard(mutex_);

  // All worker threads (and the main thread) have a ThreadInfo
  for (auto i = 0; i < numThreads; ++i) {
    threadInfo_.emplace_back(
        MakeAligned<ThreadInfo>::make(kCacheLineSize, i, numThreads));
  }

  // The first ThreadInfo is for the main thread
  for (auto i = 1; i < numThreads; ++i) {
    auto pInfo = &(threadInfo_[i]);
    auto fn = [pInfo, this, i]() { (*pInfo)->threadMain(i, this); };

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

int ThreadPool::getNumThreads() const {
  std::lock_guard<std::mutex> guard(executionMutex_);

  return threadInfo_.size();
}

// Sets the minimum work size (range) for which to invoke the
// threadpool; work sizes smaller than this will just be run on the
// main (calling) thread
void ThreadPool::setMinWorkSize(size_t size) {
  std::lock_guard<std::mutex> guard(executionMutex_);

  minWorkSize_ = size;
}

#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
void ThreadPool::setImbalanceRatio(float ratio) {
  std::lock_guard<std::mutex> guard(executionMutex_);

  imbalanceRatio_ = ratio;
}
#endif

#ifdef CAFFE2_THREADPOOL_STATS
std::vector<ThreadStats> ThreadPool::getStats(bool reset) {
  std::lock_guard<std::mutex> guard(executionMutex_);

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

    // The above serves as a barrier to ensure the stats are complete

    std::vector<ThreadStats> stats;
    for (auto& t : threadInfo_) {
      stats.push_back(t->stats_);
      if (reset) {
        t->stats_.reset();
      }
    }

    return stats;
  }
}
#endif

void ThreadPool::run(const std::function<void(int, size_t)>& fn, size_t range) {
  std::lock_guard<std::mutex> guard(executionMutex_);

  // If there are no worker threads, or if the range is too small (too
  // little work), just run locally
  bool runLocally = threads_.empty() || range < minWorkSize_ ||
      FLAGS_caffe2_threadpool_force_inline;

  auto numThreads = threadInfo_.size();
  size_t workUnitsPerThread = 0;
  size_t firstThreadWork = 0;
  size_t otherThreadWork = 0;

  if (!runLocally) {
    size_t workUnitsPerThread = (numThreads + range - 1) / numThreads;

// On mobile devices (especially big.LITTLE cores), there is
// significant lag in getting other threads to participate versus
// the current thread, which is likely already running on a big
// core.
// Based on tests, the main thread will execute (through its own
// work and stealing others) about 25% more work than other
// threads.
// To reduce the work stealing overhead, give the main thread 25%
// more work to start with.
#ifdef CAFFE2_THREADPOOL_MAIN_IMBALANCE
    firstThreadWork = (size_t)(imbalanceRatio_ * workUnitsPerThread);
    if (firstThreadWork >= range) {
      // give all to first thread
      runLocally = true;
    }

    size_t remainderWork = range - firstThreadWork;
    otherThreadWork = ((numThreads - 1) + remainderWork - 1) / (numThreads - 1);
#else
    firstThreadWork = workUnitsPerThread;
    otherThreadWork = workUnitsPerThread;
#endif
  }

  if (runLocally) {
    // Work is small enough to just run locally; multithread overhead
    // is too high
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }

#ifdef CAFFE2_THREADPOOL_STATS
    // The main thread worked on this directly
    auto& stats = threadInfo_[0]->stats_;
    stats.numWorkedOn += range;
    stats.numAssigned += range;
#endif

    return;
  }

  // Otherwise, all worker threads participate
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

    // Work given to main thread
    {
      auto& info = threadInfo_[0];
      info->rangeStart_ = 0;
      // already guaranteed to be within bounds
      info->rangeEnd_ = firstThreadWork;
      info->rangeLength_ = firstThreadWork;
#ifdef CAFFE2_THREADPOOL_STATS
      info->stats_.numAssigned += firstThreadWork;
#endif
    }

    // Work given to other threads
    size_t workStart = firstThreadWork;
    for (size_t i = 1; i < numThreads; ++i) {
      auto& info = threadInfo_[i];

      auto start = std::min(workStart, range);
      auto end = std::min(workStart + otherThreadWork, range);
      auto numAssigned = end - start;
      info->rangeStart_ = start;
      info->rangeEnd_ = end;
      info->rangeLength_ = numAssigned;
#ifdef CAFFE2_THREADPOOL_STATS
      info->stats_.numAssigned += numAssigned;
#endif
      workStart += otherThreadWork;
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

void ThreadInfo::threadMain(int threadId, ThreadPool* pool) {
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

bool ThreadInfo::runAndSteal(int threadId, ThreadPool* pool) {
  auto lambdaFunctionToRun = pool->fn_;
  int localItemsCompleted = 0;
  int localItemsStolen = 0;

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

  // Done, now look for other threads' items to steal
  for (auto i = (threadId_ + 1) % numThreads_; i != threadId_;
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
#ifdef CAFFE2_THREADPOOL_STATS
      ++localItemsStolen;
#endif
    }
  }

  bool lastThread = false;

  if (localItemsCompleted > 0) {
    auto numRemaining =
        (pool->workItemsPending_ -= localItemsCompleted); // atomic
    DCHECK_GE(numRemaining, 0);

    if (numRemaining == 0) {
      // We were the last thread to finish all work
      lastThread = true;
    }
  }

#ifdef CAFFE2_THREADPOOL_STATS
  stats_.numWorkedOn += localItemsCompleted;
  stats_.numStolen += localItemsStolen;
#endif

  return lastThread;
}

} // namespace caffe2

#endif // CAFFE2_THREADPOOL_MOBILE
