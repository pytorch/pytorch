#include "caffe2/utils/threadpool/ThreadPool.h"
#include "WorkersPool.h"
#include "caffe2/core/logging.h"

#include <cpuinfo.h>

C10_DEFINE_bool(
    caffe2_threadpool_force_inline,
    false,
    "Force to always run jobs on the calling thread");

// Whether or not threadpool caps apply to Android
C10_DEFINE_int(caffe2_threadpool_android_cap, true, "");

// Whether or not threadpool caps apply to iOS
C10_DEFINE_int(caffe2_threadpool_ios_cap, true, "");

C10_DEFINE_int(pthreadpool_size, 0, "Override the default thread pool size.");

namespace caffe2 {

size_t getDefaultNumThreads() {
  CAFFE_ENFORCE(cpuinfo_initialize(), "cpuinfo initialization failed");
  int numThreads = cpuinfo_get_processors_count();

  bool applyCap = false;
#if defined(C10_ANDROID)
  applyCap = FLAGS_caffe2_threadpool_android_cap;
#elif defined(C10_IOS)
  applyCap = FLAGS_caffe2_threadpool_ios_cap;
#endif

  if (applyCap) {
    switch (numThreads) {
#if defined(C10_ANDROID) && (CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64)
      case 4:
        switch (cpuinfo_get_core(0)->midr & UINT32_C(0xFF00FFF0)) {
          case UINT32_C(0x51002110): /* Snapdragon 820 Kryo Silver */
          case UINT32_C(0x51002010): /* Snapdragon 821 Kryo Silver */
          case UINT32_C(0x51002050): /* Snapdragon 820/821 Kryo Gold */
            /* Kryo: 2+2 big.LITTLE */
            numThreads = 2;
            break;
          default:
            /* Anything else: assume homogeneous architecture */
            numThreads = 4;
            break;
        }
        break;
#endif
      case 5:
        /* 4+1 big.LITTLE */
        numThreads = 4;
        break;
      case 6:
        /* 2+4 big.LITTLE */
        numThreads = 2;
        break;
      case 8:
        /* 4+4 big.LITTLE */
        numThreads = 4;
        break;
      case 10:
        /* 4+4+2 Min.Med.Max, running on Med cores */
        numThreads = 4;
        break;
      default:
        if (numThreads > 4) {
          numThreads = numThreads / 2;
        }
        break;
    }
  }

  if (FLAGS_pthreadpool_size) {
    // Always give precedence to explicit setting.
    numThreads = FLAGS_pthreadpool_size;
  }
  return numThreads;
}

// Default smallest amount of work that will be partitioned between
// multiple threads; the runtime value is configurable
constexpr size_t kDefaultMinWorkSize = 1;

size_t ThreadPool::defaultNumThreads_ = 0;

std::unique_ptr<ThreadPool> ThreadPool::defaultThreadPool() {
  defaultNumThreads_ = getDefaultNumThreads();
  LOG(INFO) << "Constructing thread pool with " << defaultNumThreads_
            << " threads";
  return std::make_unique<ThreadPool>(defaultNumThreads_);
}

ThreadPool::ThreadPool(int numThreads)
    : minWorkSize_(kDefaultMinWorkSize),
      numThreads_(numThreads),
      workersPool_(std::make_shared<WorkersPool>()) {}

ThreadPool::~ThreadPool() {}

int ThreadPool::getNumThreads() const {
  return numThreads_;
}

// Sets the number of threads
// # of threads should not be bigger than the number of big cores
void ThreadPool::setNumThreads(size_t numThreads) {
  if (defaultNumThreads_ == 0) {
    defaultNumThreads_ = getDefaultNumThreads();
  }
  numThreads_ = std::min(numThreads, defaultNumThreads_);
}

// Sets the minimum work size (range) for which to invoke the
// threadpool; work sizes smaller than this will just be run on the
// main (calling) thread
void ThreadPool::setMinWorkSize(size_t size) {
  std::lock_guard<std::mutex> guard(executionMutex_);
  minWorkSize_ = size;
}

void ThreadPool::run(const std::function<void(int, size_t)>& fn, size_t range) {
  const auto numThreads = numThreads_.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> guard(executionMutex_);
  // If there are no worker threads, or if the range is too small (too
  // little work), just run locally
  const bool runLocally = range < minWorkSize_ ||
      FLAGS_caffe2_threadpool_force_inline || (numThreads == 0);
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
    ~FnTask() override{};
    const std::function<void(int, size_t)>* fn_;
    int idx_;
    size_t start_;
    size_t end_;
    void Run() override {
      for (auto i = start_; i < end_; ++i) {
        (*fn_)(idx_, i);
      }
    }
  };

  CAFFE_ENFORCE_GE(numThreads_, 1);
  const size_t unitsPerTask = (range + numThreads - 1) / numThreads;
  tasks_.resize(numThreads);
  for (size_t i = 0; i < numThreads; ++i) {
    if (!tasks_[i]) {
      tasks_[i].reset(new FnTask());
    }
    auto* task = (FnTask*)tasks_[i].get();
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
  CAFFE_ENFORCE_LE(tasks_.size(), numThreads);
  CAFFE_ENFORCE_GE(tasks_.size(), 1);
  workersPool_->Execute(tasks_);
}

void ThreadPool::withPool(const std::function<void(WorkersPool*)>& f) {
  std::lock_guard<std::mutex> guard(executionMutex_);
  f(workersPool_.get());
}

} // namespace caffe2
