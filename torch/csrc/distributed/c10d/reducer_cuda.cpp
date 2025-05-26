#include <torch/csrc/distributed/c10d/reducer_timer.hpp>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>

namespace c10d {
namespace {

const int kMilliSecondToNanosSecond = 1000000;

class CudaTimer : public Timer {
 private:
  c10::Device device;

  at::cuda::CUDAEvent forward_start = at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_end =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_end = at::cuda::CUDAEvent(cudaEventDefault);

  at::cuda::CUDAEvent& getEvent(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start;
      case Event::kBackwardComputeStart:
        return backward_compute_start;
      case Event::kBackwardComputeEnd:
        return backward_compute_end;
      case Event::kBackwardCommStart:
        return backward_comm_start;
      case Event::kBackwardCommEnd:
        return backward_comm_end;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }

 public:
  explicit CudaTimer(c10::Device dev) : device(dev) {}

  void record(Event event) override {
    // Parent class sets the host-side time
    Timer::record(event);
    c10::DeviceGuard g(device);
    getEvent(event).record();
  }

  std::optional<int64_t> measureDifference(Event start, Event end) override {
    c10::DeviceGuard g(device);
    at::cuda::CUDAEvent& start_event = getEvent(start);
    at::cuda::CUDAEvent& end_event = getEvent(end);
    // It is possible users did not call backward or run codes in
    // no-sync mode, in this case, some cudaEvents like "backward_compute_end"
    // or "backward_comm_start" or "backward_comm_end" will not be recorded.
    // cudaEvent is created when it is first time to be recorded.
    // If it is never recorded/created, skip synchronize and calculation.
    // Otherwise it will throw cuda errors.
    if (!start_event.isCreated() || !end_event.isCreated()) {
      return std::nullopt;
    }
    // set_runtime_stats_and_log is called at the beginning of forward call,
    // when it is cheap to synchronize the cuda events of previous iteration,
    // as mostly all cuda operations are finished in previous iteration.
    start_event.synchronize();
    end_event.synchronize();
    float milliseconds = start_event.elapsed_time(end_event);
    // If gpu_end is not recorded in this iteration,
    // milliseconds will have invalid value.
    // For some cases like DDP runs on non-sync mode,
    // gpu_end can not be recorded in this iteration and thus can not
    // calculate the valid avg_time.
    // In this case, skip calculating the avg_time and return.
    if (milliseconds < 0) {
      return std::nullopt;
    }
    return int64_t(milliseconds * kMilliSecondToNanosSecond);
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCUDA, CudaTimer)

} // namespace
} // namespace c10d
