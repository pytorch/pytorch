#include <condition_variable>
#include <mutex>
#include <stack>

#if !defined(_MSC_VER) && !defined(__APPLE__)
#include <sched.h>
#endif

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/net_simple.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

namespace gpu_single_thread {

struct Task {
  std::vector<std::unique_ptr<OperatorBase>>* ops_;
  std::condition_variable* cv_;
  std::mutex* mtx_;
  int stream_id_;
  bool done_ = false;
};

class GPUExecutor {
 public:
  explicit GPUExecutor(int gpu_id) : gpu_id_(gpu_id) {}

  ~GPUExecutor() {
    queue_.NoMoreJobs();
    thread_.join();
  }

  void RunJob(Task* task) {
    queue_.Push(task);
  }

  void start() {
    thread_ = std::thread(&GPUExecutor::WorkerFunction, this);
  }

  static std::shared_ptr<GPUExecutor> Get(int gpu);
  static void Release(int gpu);

 private:
  void set_affinity();
  void WorkerFunction();

  std::thread thread_;
  int gpu_id_;
  SimpleQueue<Task*> queue_;
  static std::shared_ptr<GPUExecutor> executors_[CAFFE2_COMPILE_TIME_MAX_GPUS];
  static std::mutex gpu_mtx_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};

std::shared_ptr<GPUExecutor>
    GPUExecutor::executors_[CAFFE2_COMPILE_TIME_MAX_GPUS];
std::mutex GPUExecutor::gpu_mtx_[CAFFE2_COMPILE_TIME_MAX_GPUS];

std::shared_ptr<GPUExecutor> GPUExecutor::Get(int gpu) {
  std::lock_guard<std::mutex> grd(gpu_mtx_[gpu]);
  if (!executors_[gpu].get()) {
    executors_[gpu].reset(new GPUExecutor(gpu));
    executors_[gpu].get()->start();
  }
  return executors_[gpu];
}

void GPUExecutor::Release(int gpu) {
  std::lock_guard<std::mutex> grd(gpu_mtx_[gpu]);
  if (executors_[gpu].use_count() == 1) {
    executors_[gpu].reset();
  }
}

void GPUExecutor::set_affinity() {
// TODO: find a Windows-compatible affinity setting approach.
// Currently, set_affinity has no effect in Windows. The code is still
// correct with possible slowdowns.
#if !defined(_MSC_VER) && !defined(__APPLE__)
  /* Set CPU affinity */
  int num_cores = std::thread::hardware_concurrency();
  if (num_cores > 0) {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    CPU_SET(gpu_id_ % num_cores, &mask);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &mask)) {
      LOG(WARNING) << "Could not set CPU affinity";
    }
  }
#endif
}

// Worker that takes list of operators from the queue
// and executes them.
void GPUExecutor::WorkerFunction() {
  int stream_id_seq = 0;
  std::stack<int> streams;
  set_affinity();

  while (true) {
    Task* task = nullptr;
    vector<Task*> task_batch;

    if (!queue_.Pop(&task)) {
      return;
    }
    int num_tasks = 1 + queue_.size();

    // Grab all tasks currently in queue so we can run them in parallel
    // Since we have only one producer, we know this does not block

    // TODO: launch ops in "zig-zag" manner so that we can start multiple
    // streams as simultaneously as possible
    for (int i = num_tasks - 1; i >= 0; i--) {
      assert(task != nullptr);
      if (streams.empty()) {
        task->stream_id_ = stream_id_seq++;
      } else {
        task->stream_id_ = streams.top();
        streams.pop();
      }

      for (auto& op : *task->ops_) {
        op->RunAsync(task->stream_id_);
      }
      task_batch.push_back(task);

      // Get the next one
      if (i > 0) {
        if (!queue_.Pop(&task)) {
          return;
        }
      }
    }

    // Wait for the currently executing streams
    for (auto& pendtask : task_batch) {
      cudaStream_t stream =
          CUDAContext::cuda_stream(gpu_id_, pendtask->stream_id_);
      CUDA_ENFORCE(cudaStreamSynchronize(stream));
      streams.push(pendtask->stream_id_);
      std::unique_lock<std::mutex> lk(*pendtask->mtx_);
      pendtask->done_ = true;
      pendtask->cv_->notify_one();
    }
  }
}

class SingleThreadAsyncNet : public SimpleNet {
 public:
  using SimpleNet::SimpleNet;

  ~SingleThreadAsyncNet() {
    if (executor_.get()) {
      // Explicitly reset my holding of the exeuctor so it can be
      // killed.
      executor_.reset();
      GPUExecutor::Release(gpu_id_);
    }
  }

  bool Run() override {
    if (!executor_.get()) {
      initialize();
    }

    // Dispatch jobs to the gpu-specific executor thread
    std::unique_lock<std::mutex> lk(mutex_);
    Task t;
    t.ops_ = &operators_;
    t.cv_ = &cv_;
    t.mtx_ = &mutex_;
    t.done_ = false;
    executor_.get()->RunJob(&t);

    while (!t.done_) {
      cv_.wait(lk);
    }

    return true;
  }

 private:
  std::condition_variable cv_;
  std::mutex mutex_;

  void initialize() {
    std::lock_guard<std::mutex> grd(mutex_);

    /* Check the gpu id of this net and check that only one
       GPU has operators on this net */
    gpu_id_ = (-1);
    for (auto& op : operators_) {
      if (op->device_option().device_type() == CUDA) {
        if (gpu_id_ < 0) {
          gpu_id_ = op->device_option().cuda_gpu_id();
        } else {
          CAFFE_ENFORCE_EQ(
              gpu_id_,
              op->device_option().cuda_gpu_id(),
              "One net can only have operators for one GPU");
        }
      }
    }
    executor_ = GPUExecutor::Get(gpu_id_);
  }

  int gpu_id_;
  std::shared_ptr<GPUExecutor> executor_;
};

REGISTER_NET(singlethread_async, SingleThreadAsyncNet)

} // namespace gpu_single_thread
} // namespace caffe2
