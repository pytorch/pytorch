
#include <ATen/native/ForeachOps.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

std::vector<Tensor> _foreach_add(TensorList tensors, Scalar scalar) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list can't be empty.");
  }

  for (int i = 0; i < tensors.size(); i++) {
    if (tensors[i].layout() != at::kStrided) {
      throw std::runtime_error("Only tensors with strided layouts are supported.");
    }

    if (!tensors[i].is_non_overlapping_and_dense()) {
      throw std::runtime_error("Only non overlapping and dense tensors are supported.");
    }
  }
  return foreach_tensor_add_scalar_stub(tensors[0].device().type(), tensors, scalar);
}
DEFINE_DISPATCH(foreach_tensor_add_scalar_stub);

}}
