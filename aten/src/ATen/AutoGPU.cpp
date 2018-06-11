#include <ATen/AutoGPU.h>

#include <ATen/Tensor.h>
#include <ATen/TensorMethods.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {
// This constructor is only in its own .cpp file because the nvcc device
// compiler cannot compile the use of `optional`.
AutoGPU::AutoGPU(optional<int32_t> index) {
  set_index(index);
}

void AutoGPU::set_index_from(const Tensor& tensor) {
  if (tensor.type().is_cuda()) {
    set_index(tensor.get_device());
  }
}
} // namespace at
