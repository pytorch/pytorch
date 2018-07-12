#include <torch/cuda.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace cuda {
size_t device_count() {
  return at::globalContext().getNumGPUs();
}

bool is_available() {
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  return cuda::device_count() > 0;
}

bool cudnn_is_available() {
  return is_available() && at::globalContext().hasCuDNN();
}
} // namespace cuda
} // namespace torch
