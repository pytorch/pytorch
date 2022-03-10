#include <ATen/ATen.h>

namespace at {
namespace native {
TORCH_API Tensor
quantized_add(Tensor qa, Tensor qb, double scale, int64_t zero_point);
}
} // namespace at
