#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

Tensor cov(
    const Tensor& self,
    int64_t correlation,
    const c10::optional<Tensor>& fweights,
    const c10::optional<Tensor>& aweights) {

  // Check Preconditions
  checkDimRange("cov", TensorArg{self, "input", 1}, 0, 2);

  return at::empty(0);
}

} // namespace native
} // namespace at