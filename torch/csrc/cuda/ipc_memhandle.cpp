#include <torch/csrc/cuda/ipc_memhandle.h>

#include <unordered_map>
#include <mutex>

namespace torch {
  namespace {
    std::mutex IpcMemMutex;
    std::unordered_map<std::string, void*> ipcMemHandle_to_devptr;
  }

  void* getCachedCUDAIpcDevptr(std::string handle) {
    std::lock_guard<std::mutex> guard(IpcMemMutex);
    if (ipcMemHandle_to_devptr.find(handle) == ipcMemHandle_to_devptr.end()) {
      void *devPtr = nullptr;
      cudaIpcMemHandle_t ipc_handle = *(cudaIpcMemHandle_t*)handle.c_str();
      THCudaCheck(cudaIpcOpenMemHandle(&devPtr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));
      ipcMemHandle_to_devptr[handle] = devPtr;
    }
    return ipcMemHandle_to_devptr[handle];
  }
} // namespace torch
