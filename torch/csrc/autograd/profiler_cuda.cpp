#include <torch/csrc/autograd/profiler.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>

#include <sstream>

namespace torch { namespace autograd { namespace profiler {

namespace {

static inline void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == cudaErrorInitializationError) {
      // It is common for users to use DataLoader with multiple workers
      // and the autograd profiler. Throw a nice error message here.
      ss << "CUDA initialization error. "
         << "This can occur if one runs the profiler in CUDA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in CUDA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      ss << cudaGetErrorString(result);
    }
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CUDA_CHECK(result) cudaCheck(result,__FILE__,__LINE__);

struct CUDAMethods : public CUDAStubs {
  void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) const override {
      LOG(INFO) << "Start record";
    TORCH_CUDA_CHECK(cudaGetDevice(device));
    // Hack: setting device
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(i);
    LOG(INFO) << "Done setting device";
    CUevent_st* cuda_event_ptr;
    TORCH_CUDA_CHECK(cudaEventCreate(&cuda_event_ptr));
    *event = std::shared_ptr<CUevent_st>(cuda_event_ptr, [](CUevent_st* ptr) {
      TORCH_CUDA_CHECK(cudaEventDestroy(ptr));
    });
    auto stream = at::cuda::getCurrentCUDAStream();
    *cpu_ns = getTime();
    TORCH_CUDA_CHECK(cudaEventRecord(cuda_event_ptr, stream));
    LOG(INFO) << "Done with record";
  }

  float elapsed(const CUDAEventStub* event, const CUDAEventStub* event2) const override{
      LOG(INFO) << "Calling cudaEventSynchronize";
    TORCH_CUDA_CHECK(cudaEventSynchronize(event->get()));
    TORCH_CUDA_CHECK(cudaEventSynchronize(event2->get()));
    LOG(INFO) << "Synchronized event and event2";
    float ms;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, event->get(), event2->get()));
    return ms*1000.0;
  }

  void nvtxMarkA(const char* name) const override {
    ::nvtxMark(name);
  }

  void nvtxRangePushA(const char* name) const override {
    ::nvtxRangePushA(name);
  }

  void nvtxRangePop() const override {
    ::nvtxRangePop();
  }

  void onEachDevice(std::function<void(int)> op) const override {
      LOG(INFO) << "start on each device";
    at::cuda::OptionalCUDAGuard device_guard;
    int count = at::cuda::device_count();
    for(int i = 0; i < count; i++) {
      device_guard.set_index(i);
      op(i);
    }
    LOG(INFO) << "Done with onEachDevice";
  }

  void synchronize() const override {
      LOG(INFO) << "Calling cudaDeviceSynchronize";
    cudaDeviceSynchronize();
    LOG(INFO) << "Done with cudaDeviceSynchronize";
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterCUDAMethods {
  RegisterCUDAMethods() {
    static CUDAMethods methods;
    registerCUDAMethods(&methods);
  }
};
RegisterCUDAMethods reg;

} // namespaces
} // namespace profiler
} // namespace autograd
} // namespace torch
