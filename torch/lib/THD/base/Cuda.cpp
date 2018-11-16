#include "Cuda.hpp"
#include <unordered_map>

#ifdef USE_CUDA
THCState** _THDCudaState;

void THDSetCudaStatePtr(THCState** state) {
  _THDCudaState = state;
}

static int nextStreamId = 1; // 0 for the default stream
static std::unordered_map<cudaStream_t, int> streamIdMap;

void THDRegisterCudaStream(cudaStream_t stream) {
  streamIdMap.emplace(stream, nextStreamId++);
}

int THDGetStreamId(cudaStream_t stream) {
  if (!stream)
    return 0;
  auto it = streamIdMap.find(stream);
  if (it == streamIdMap.end()) {
    throw std::runtime_error(
        "using a stream that's hasn't been registered in THD");
  }
  return it->second;
}
#endif
