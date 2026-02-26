#include <ATen/core/Tensor.h>

namespace at::native::internal {

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cuda_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts);

} // namespace at::native::internal
