#include <torch/csrc/cuda/ipc_memhandle.h>

#include <THC/THC.h>
#include <unordered_map>
#include <mutex>

namespace torch {
  namespace {
    std::mutex IpcMemMutex;
    std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
  }

  std::shared_ptr<void> getCachedCUDAIpcDevptr(std::string handle) {
    std::lock_guard<std::mutex> guard(IpcMemMutex);
    std::shared_ptr<void> sp;
    if (ipcMemHandle_to_devptr.find(handle) == ipcMemHandle_to_devptr.end()
        || ipcMemHandle_to_devptr[handle].expired()) {
      void *devPtr = nullptr;
      cudaIpcMemHandle_t ipc_handle = *(cudaIpcMemHandle_t*)handle.c_str();
      THCudaCheck(cudaIpcOpenMemHandle(&devPtr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));
      std::cout << "opened devPtr " << devPtr << std::endl;
      sp = std::shared_ptr<void>(devPtr, [](void *ptr) {THCudaCheck(cudaIpcCloseMemHandle(ptr));});
      std::cout << "get from shared_ptr " << sp.get() << std::endl;
      std::weak_ptr<void> wp = sp;
      ipcMemHandle_to_devptr[handle] = wp;
    }
    auto res = ipcMemHandle_to_devptr[handle].lock();
    std::cout << "from map: " << res.get() << std::endl;
    return res;
  }
} // namespace torch
