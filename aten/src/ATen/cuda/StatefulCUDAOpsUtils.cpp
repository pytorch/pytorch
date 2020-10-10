#include <StatefulCUDAOpsUtils.cuh>

namespace at {
namespace cuda {

namespace {
// Maintains a per-device array of state update streams.
// Imitates the pattern for per-device generators in CUDAGeneratorImpl.cpp.

// Ensures we only call cudaGetDeviceCount only once.
std::once_flag state_streams_init_flag;

// Total number of gpus in the system.
int64_t num_gpus;

// Ensures state_streams is initialized once.
std::deque<std::once_flag> state_streams_init_flags;

// Default state streams, one per GPU
std::vector<c10::optional<c10::Stream>> state_streams;

/*
* Populates the global variables related to CUDA generators
* Warning: this function must only be called once!
*/
void initCUDAGenVector(){
  num_gpus = c10::cuda::device_count();
  state_streams_init_flags.resize(num_gpus);
  state_streams.reserve(num_gpus);
}
} // anonymous namespace

c10::optional<c10::Stream> stateUpdateStream(DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(at::globalContext().statefulCUDAOpStatesOnDevice());
  std::call_once(state_streams_init_flag, initCUDAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  CUDAGuard device_guard(idx);
  std::call_once(state_streams_init_flags[idx],
                 [&] {
                   state_streams[idx] = getStreamFromPool(/*isHighPriority=*/true,
                                                          /*index=*/idx);
                 });
  return state_streams[idx];
}

} // namespace cuda
} // namespace at
