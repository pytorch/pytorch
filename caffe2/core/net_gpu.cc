#include "caffe2/core/net.h"

#include "caffe2/core/context_gpu.h"
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
    CHECK(range_);
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

struct Stream;

struct Event {
 public:
  explicit Event(const DeviceOption& device_option) {
    if (device_option.device_type() == CUDA) {
      gpu_id_ = device_option.has_cuda_gpu_id() ? device_option.cuda_gpu_id()
                                                : GetDefaultGPUID();
      DeviceGuard g(gpu_id_);
      CUDA_CHECK(cudaEventCreateWithFlags(
          &event_, cudaEventDefault | cudaEventDisableTiming));
    }
  }

  ~Event() {
    if (event_) {
      CUDA_CHECK(cudaEventDestroy(event_));
    }
  }

  void record(const Stream& stream);

  int gpu_id_{-1};
  cudaEvent_t event_{nullptr};
  bool outstanding_{false};
  bool neverRecorded_{true};
  DISABLE_COPY_AND_ASSIGN(Event);
};

struct Stream {
  explicit Stream(const DeviceOption& device_option) {
    if (device_option.device_type() == CUDA) {
      gpu_id_ = device_option.has_cuda_gpu_id() ? device_option.cuda_gpu_id()
                                                : GetDefaultGPUID();
      stream_ = CHECK_NOTNULL(CUDAContext::cuda_stream(gpu_id_));
    }
  }

  void wait(Event* event) const {
    CHECK(event);
    event->outstanding_ = false;
    if (!event->event_) {
      return;
    }

    if (!stream_) {
      CHECK_EQ(gpu_id_, -1);
      CUDA_CHECK(cudaEventSynchronize(event->event_));
      return;
    }

    CHECK_NE(gpu_id_, -1);
    VLOG_IF(2, gpu_id_ != event->gpu_id_) << "Cross-device waiting: " << gpu_id_
                                          << " waiting on " << event->gpu_id_;
    DeviceGuard g(gpu_id_);
    CUDA_CHECK(cudaStreamWaitEvent(stream_, event->event_, 0));
  }

  int gpu_id_{-1};
  cudaStream_t stream_{nullptr};

 private:
  DISABLE_COPY_AND_ASSIGN(Stream);
};

void Event::record(const Stream& stream) {
  if (outstanding_) {
    // TODO - should we do this?
    stream.wait(this);
  }
  CHECK(!outstanding_) << "Failed to wait on event before recording";
  CHECK_EQ(stream.gpu_id_, gpu_id_);
  // We *never* use the default stream in Caffe2, so stream should
  // never be NULL for a compute stream in Caffe2.
  if (!stream.stream_) {
    CHECK(!event_);
    return;
  }

  CHECK(event_);
  DeviceGuard g(gpu_id_);
  CUDA_CHECK(cudaEventRecord(event_, stream.stream_));
  outstanding_ = true;
}
}

// Run an event-driven graph - before each operator chain, wait on
// each parent operator for the chain source (Stream::wait), then
// execute each operator (implicitly on the same stream).
class AsyncDAGNet : public DAGNetBase {
 public:
  AsyncDAGNet(const NetDef& net_def, Workspace* ws) : DAGNetBase(net_def, ws) {
    eventRecorded_.resize(net_def.op_size());
    events_.reserve(net_def.op_size());
    for (int idx = 0; idx < net_def.op_size(); ++idx) {
      const OperatorDef& op_def = net_def.op(idx);
      if (!op_def.has_device_option() && net_def.has_device_option()) {
        OperatorDef temp_def(op_def);
        temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
        events_.emplace_back(new Event(temp_def.device_option()));
      } else {
        events_.emplace_back(new Event(op_def.device_option()));
      }
    }
  }

  bool RunAt(const std::vector<int>& chain) override {
    CHECK(!chain.empty());
    const auto source_idx = chain.front();
    Stream stream{operator_nodes_[source_idx].operator_->def().device_option()};
    const auto& parents = operator_nodes_[source_idx].parents_;
    // Help ensure that our chaining is correct by verifying at least
    // one parent recorded an event.
    CHECK(
        parents.empty() ||
        std::any_of(parents.begin(), parents.end(), [this](int p) {
          return eventRecorded_[p];
        }));

    for (auto source_parent_idx : operator_nodes_[source_idx].parents_) {
      ProfiledRange r(
          operator_nodes_[source_parent_idx].operator_->def(), kWaitColor);
      stream.wait(events_[source_parent_idx].get());
    }

    // We've waited on all our parent indices.
    bool success = true;
    for (auto idx : chain) {
      ProfiledRange r(operator_nodes_[idx].operator_->def(), kRunColor);
      success &= operator_nodes_[idx].operator_->RunAsync();
    }

    // Record an event for the sink of the chain.
    const auto& sink_idx = chain.back();
    {
      ProfiledRange r(operator_nodes_[sink_idx].operator_->def(), kRecordColor);
      events_[sink_idx]->record(stream);
    }
    CHECK(!eventRecorded_[sink_idx]);
    eventRecorded_[sink_idx] = 1;
    return success;
  }

  bool Run() override {
    // Reset the event tracking at each iteration
    eventRecorded_.assign(eventRecorded_.size(), 0);

    const auto result = DAGNetBase::Run();

    // Synchronize execution of the network with respect to the host.
    DeviceOption device_option;
    device_option.set_device_type(CPU);
    Stream stream{device_option};

    // Potential optimization: we can pre-compute outstanding events.
    for (auto i = 0; i < events_.size(); ++i) {
      auto& event = events_[i];
      if (event->outstanding_) {
        VLOG(2) << "Synchronizing host on outstanding event";
        ProfiledRange r(operator_nodes_[i].operator_->def(), kWaitColor);
        stream.wait(event.get());
      }
    }
    return result;
  }

 protected:
  // Tracks whether a given op has had an event recorded in each
  // RunAt() iteration.

  std::vector<int32_t> eventRecorded_;
  std::vector<std::unique_ptr<Event>> events_;
  DISABLE_COPY_AND_ASSIGN(AsyncDAGNet);
};

REGISTER_NET(async_dag, AsyncDAGNet);
}
