#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h> // Need that for the `native_functions.yaml`
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
  namespace native {
    Tensor& zero_quantized_cpu_(Tensor &self) {
    return self.fill_(0);
    }
}}  // namespace at::native
