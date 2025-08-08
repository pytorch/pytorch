#include <ATen/core/Tensor.h>

namespace at::native {
  std::tuple<at::Tensor, at::Tensor> attention(const at::Tensor & query,
                                      const at::Tensor & key,
                                      const at::Tensor & value
                                      );
} // namespace at::native
