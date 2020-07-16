
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
  }
  return foreach_tensor_add_scalar_stub(tensors[0].device().type(), tensors, scalar);
}
DEFINE_DISPATCH(foreach_tensor_add_scalar_stub);

}}
