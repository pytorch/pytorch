#include <ATen/Tensor.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/Constants.h>
#include <functorch/csrc/DynamicLayer.h>

namespace at { namespace functorch {

Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level);
std::tuple<Tensor, optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level);

}}

