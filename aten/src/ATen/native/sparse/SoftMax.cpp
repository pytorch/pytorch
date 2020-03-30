#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {
namespace {
} // namespace

  Tensor softmax_sparse_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
    AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
    auto input = input_;
    Tensor output = at::native::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    if (input.numel() == 0) {
      return output;
    }
    return output;
  }

}
}
