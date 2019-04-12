#include <torch/csrc/utils/tensor_int_repr.h>

namespace torch { namespace utils {

at::Tensor int_repr(at::Tensor t) {
  return t.to(t.device(), at::kByte);
}

}} // namespace torch::utils
