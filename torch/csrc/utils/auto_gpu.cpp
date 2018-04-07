#include <torch/csrc/utils/auto_gpu.h>

#include <ATen/ATen.h>

#include <sstream>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace {
void cudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::string msg = "CUDA error (";
    msg += std::to_string(err);
    msg += "): ";
    msg += cudaGetErrorString(err);
    AT_ERROR("%s", msg.c_str());
  }
}
} // namespace

namespace torch {
AutoGPU::AutoGPU(int device) {
  setDevice(device);
}

AutoGPU::AutoGPU(const at::Tensor& t) {
  setDevice(t.type().is_cuda() ? (int)t.get_device() : -1);
}

AutoGPU::AutoGPU(at::TensorList& tl) {
  if (tl.size() > 0) {
    auto& t = tl[0];
    setDevice(t.type().is_cuda() ? t.get_device() : -1);
  }
}

AutoGPU::~AutoGPU() {
#ifdef WITH_CUDA
  if (original_device != -1) {
    cudaSetDevice(original_device);
  }
#endif
}

void AutoGPU::setDevice(int device) {
#ifdef WITH_CUDA
  if (device == -1) {
    return;
  }
  if (original_device == -1) {
    cudaCheck(cudaGetDevice(&original_device));
    if (device != original_device) {
      cudaCheck(cudaSetDevice(device));
    }
  } else {
    cudaCheck(cudaSetDevice(device));
  }
#endif
}
} // namespace torch
