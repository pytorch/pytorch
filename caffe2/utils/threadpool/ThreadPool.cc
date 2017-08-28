#include "caffe2/utils/threadpool/ThreadPool.h"
#include "WorkersPool.h"
#include "caffe2/core/logging.h"

#if CAFFE2_ANDROID
#include <cpu-features.h>
#endif

CAFFE2_DEFINE_bool(caffe2_threadpool_force_inline, false,
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
      // limit to 4 - seems to work well for 2+2, 4+1, 4+0.
      numThreads = 4;
    } else {
      // Use half the cores
      numThreads = numThreads / 2;
    }
  }
  LOG(INFO) << "Constructing thread pool with " << numThreads << " threads";
  return caffe2::make_unique<ThreadPool>(numThreads);
}

ThreadPool::ThreadPool(int numThreads)
    : minWorkSize_(kDefaultMinWorkSize), numThreads_(numThreads),
      workersPool_(std::make_shared<WorkersPool>()) {}

ThreadPool::~ThreadPool() {}

int ThreadPool::getNumThreads() const {
  std::lock_guard<std::mutex> guard(executionMutex_);
  return numThreads_;
}

// Sets the minimum work size (range) for which to invoke the
// threadpool; work sizes smaller than this will just be run on the
// main (calling) thread
void ThreadPool::setMinWorkSize(size_t size) {
  std::lock_guard<std::mutex> guard(executionMutex_);
  minWorkSize_ = size;
}

void ThreadPool::run(const std::function<void(int, size_t)>& fn, size_t range) {
  std::lock_guard<std::mutex> guard(executionMutex_);
  // If there are no worker threads, or if the range is too small (too
  // little work), just run locally
  const bool runLocally = range < minWorkSize_ ||
                          FLAGS_caffe2_threadpool_force_inline ||
                          (numThreads_ == 0);
  if (runLocally) {
    // Work is small enough to just run locally; multithread overhead
    // is too high
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }
    return;
  }

  struct FnTask : public Task {
    FnTask(){};
    virtual ~FnTask(){};
    const std::function<void(int, size_t)> *fn_;
    int idx_;
    size_t start_;
    size_t end_;
    virtual void Run() override {
      for (auto i = start_; i < end_; ++i) {
        (*fn_)(idx_, i);
      }
    }
  };

  CAFFE_ENFORCE_GE(numThreads_, 1);
  const size_t unitsPerTask = (range + numThreads_ - 1) / numThreads_;
  tasks_.resize(numThreads_);
  for (size_t i = 0; i < numThreads_; ++i) {
    if (!tasks_[i]) {
      tasks_[i].reset(new FnTask());
    }
    auto *task = (FnTask *)tasks_[i].get();
    task->fn_ = &fn;
    task->idx_ = i;
    task->start_ = std::min<size_t>(range, i * unitsPerTask);
    task->end_ = std::min<size_t>(range, (i + 1) * unitsPerTask);
    if (task->start_ >= task->end_) {
      tasks_.resize(i);
      break;
    }
    CAFFE_ENFORCE_LE(task->start_, range);
    CAFFE_ENFORCE_LE(task->end_, range);
  }
  CAFFE_ENFORCE_LE(tasks_.size(), numThreads_);
  CAFFE_ENFORCE_GE(tasks_.size(), 1);
  workersPool_->Execute(tasks_);
}

} // namespace caffe2

#endif // CAFFE2_THREADPOOL_MOBILE
