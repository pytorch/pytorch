#ifndef CAFFE2_CORE_NET_GPU_H_
#define CAFFE2_CORE_NET_GPU_H_

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/net.h"

namespace caffe2 {

namespace internal {

struct Stream;

struct Event {
 public:
  explicit Event(const DeviceOption& device_option);
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

} // namespace internal

// Run an event-driven graph - before each operator chain, wait on
// each parent operator for the chain source (Stream::wait), then
// execute each operator (implicitly on the same stream).
class AsyncDAGNet : public DAGNetBase {
 public:
  AsyncDAGNet(const NetDef& net_def, Workspace* ws);
  bool RunAt(const std::vector<int>& chain) override;
  bool Run() override;

 protected:
  // Tracks whether a given op has had an event recorded in each
  // RunAt() iteration.
  std::vector<int32_t> eventRecorded_;
  std::vector<std::unique_ptr<internal::Event>> events_;
  DISABLE_COPY_AND_ASSIGN(AsyncDAGNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_GPU_H_
