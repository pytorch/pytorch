#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/IPCTypes.h>

namespace torch {
inline bool CudaIPCCollect() {
  return IPCCollect();
}
} // namespace torch

namespace c10 {
namespace {
class CudaIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch::IPCCollect();
  }
};
} // namespace
} // namespace c10
