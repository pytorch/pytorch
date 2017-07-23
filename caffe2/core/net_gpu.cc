#include "caffe2/core/net_gpu.h"

#include <condition_variable>
#include <mutex>
#include <stack>

#if !defined(_MSC_VER) && !defined(__APPLE__)
#include <sched.h>
#endif

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"

#ifdef CAFFE2_USE_NVTX
#include <nvToolsExt.h>
#endif

CAFFE2_DEFINE_bool(caffe2_use_nvtx, false, "Use NVTX ranges for profiling");

namespace caffe2 {

namespace {

using Color = int32_t;
constexpr Color kRunColor = 0x0000CCFF; // blue
constexpr Color kRecordColor = 0x00FF3300; // red
constexpr Color kWaitColor = 0x0066FF33; // green

#ifdef CAFFE2_USE_NVTX

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = def.type().c_str();
    range_ = nvtxRangeStartEx(&eventAttrib);
    CAFFE_ENFORCE(range_, "Start range is invalid.");
  }

  ~ProfiledRange() {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxRangeEnd(range_);
  }

 private:
  nvtxRangeId_t range_ = 0;
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#else

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {}

 private:
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#endif // ifdef CAFFE2_USE_NVTX

} // namespace

namespace internal {

struct Stream {
  explicit Stream(const DeviceOption& device_option) {
    if (device_option.device_type() == CUDA) {
      gpu_id_ = device_option.has_cuda_gpu_id() ? device_option.cuda_gpu_id()
                                                : GetDefaultGPUID();
      stream_ = CHECK_NOTNULL(CUDAContext::cuda_stream(gpu_id_, 0));
    }
  }

  void wait(Event* event) const {
    CAFFE_ENFORCE(event, "Event is invalid.");
    event->outstanding_ = false;
    if (!event->event_) {
      return;
    }

    if (!stream_) {
      CAFFE_ENFORCE(gpu_id_ == -1, "Gpu ID should be -1.");
      CUDA_ENFORCE(cudaEventSynchronize(event->event_));
      return;
    }

    CAFFE_ENFORCE(gpu_id_ != -1, "Gpu ID should not be -1.");
    VLOG_IF(2, gpu_id_ != event->gpu_id_) << "Cross-device waiting: " << gpu_id_
                                          << " waiting on " << event->gpu_id_;
    DeviceGuard g(gpu_id_);
    CUDA_ENFORCE(cudaStreamWaitEvent(stream_, event->event_, 0));
  }

  int gpu_id_{-1};
  cudaStream_t stream_{nullptr};

 private:
  DISABLE_COPY_AND_ASSIGN(Stream);
};

Event::Event(const DeviceOption& device_option) {
  if (device_option.device_type() == CUDA) {
    gpu_id_ = device_option.has_cuda_gpu_id() ? device_option.cuda_gpu_id()
                                              : GetDefaultGPUID();
    DeviceGuard g(gpu_id_);
    CUDA_ENFORCE(cudaEventCreateWithFlags(
        &event_, cudaEventDefault | cudaEventDisableTiming));
  }
}

void Event::record(const Stream& stream) {
  if (outstanding_) {
    // TODO - should we do this?
    stream.wait(this);
  }
  CAFFE_ENFORCE(!outstanding_, "Failed to wait on event before recording.");
  CAFFE_ENFORCE(
      stream.gpu_id_ == gpu_id_,
      "Stream gpu id ",
      stream.gpu_id_,
      " doesn't match to ",
      gpu_id_,
      ".");
  // We *never* use the default stream in Caffe2, so stream should
  // never be NULL for a compute stream in Caffe2.
  if (!stream.stream_) {
    CAFFE_ENFORCE(!event_, "Stream is NULL, so should be the event.");
    return;
  }

  CAFFE_ENFORCE(event_, "Event should not be NULL.");
  DeviceGuard g(gpu_id_);
  CUDA_ENFORCE(cudaEventRecord(event_, stream.stream_));
  outstanding_ = true;
}

} // namespace internal

AsyncDAGNet::AsyncDAGNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : DAGNetBase(net_def, ws) {
  VLOG(1) << "Constructing Async DAG Net " << net_def->name();
  eventRecorded_.resize(net_def->op_size());
  events_.reserve(net_def->op_size());
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const OperatorDef& op_def = net_def->op(idx);
    if (!op_def.has_device_option() && net_def->has_device_option()) {
      OperatorDef temp_def(op_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      events_.emplace_back(new internal::Event(temp_def.device_option()));
    } else {
      events_.emplace_back(new internal::Event(op_def.device_option()));
    }
  }
}

bool AsyncDAGNet::RunAt(const std::vector<int>& chain) {
  CAFFE_ENFORCE(!chain.empty(), "Chain should not be empty.");
  const auto source_idx = chain.front();
  internal::Stream stream{
      operator_nodes_[source_idx].operator_->device_option()};
  const auto& parents = operator_nodes_[source_idx].parents_;
  // Help ensure that our chaining is correct by verifying at least
  // one parent recorded an event.
  CAFFE_ENFORCE(
      parents.empty() || std::any_of(
                             parents.begin(),
                             parents.end(),
                             [this](int p) { return eventRecorded_[p]; }),
      "None of the parent is recorded for an event.");

  for (auto source_parent_idx : operator_nodes_[source_idx].parents_) {
    ProfiledRange r(
        operator_nodes_[source_parent_idx].operator_->debug_def(), kWaitColor);
    stream.wait(events_[source_parent_idx].get());
  }

  // We've waited on all our parent indices.
  bool success = true;
  for (auto idx : chain) {
    ProfiledRange r(operator_nodes_[idx].operator_->debug_def(), kRunColor);
    success &= operator_nodes_[idx].operator_->RunAsync();
  }

  // Record an event for the sink of the chain.
  const auto& sink_idx = chain.back();
  {
    ProfiledRange r(
        operator_nodes_[sink_idx].operator_->debug_def(), kRecordColor);
    events_[sink_idx]->record(stream);
  }
  CAFFE_ENFORCE(
      !eventRecorded_[sink_idx],
      "An event for ",
      sink_idx,
      " should not be recorded.");
  eventRecorded_[sink_idx] = 1;
  return success;
}

bool AsyncDAGNet::Run() {
  // Reset the event tracking at each iteration
  eventRecorded_.assign(eventRecorded_.size(), 0);

  const auto result = DAGNetBase::Run();

  // Synchronize execution of the network with respect to the host.
  DeviceOption device_option;
  device_option.set_device_type(CPU);
  internal::Stream stream{device_option};

  // Potential optimization: we can pre-compute outstanding events.
  for (auto i = 0; i < events_.size(); ++i) {
    auto& event = events_[i];
    if (event->outstanding_) {
      VLOG(2) << "Synchronizing host on outstanding event";
      ProfiledRange r(operator_nodes_[i].operator_->debug_def(), kWaitColor);
      stream.wait(event.get());
    }
  }
  return result;
}

REGISTER_NET(async_dag, AsyncDAGNet);

/**
  * Code for special net type that uses one executor -thread per GPU.
  */
namespace gpu_single_thread {

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

namespace {
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

  bool Run() {
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

  bool RunAsync() {
    CAFFE_THROW("RunAsync() not implemented for singlethread_async net");
    // Just to suppress compiler warning.
    return false;
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

} // namespace
} // end gpu_single_thread namespace
}
