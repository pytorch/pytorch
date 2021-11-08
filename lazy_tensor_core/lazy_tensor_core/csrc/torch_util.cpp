#include "lazy_tensor_core/csrc/torch_util.h"


namespace torch_lazy_tensors {

at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor& ref, at::ScalarType dest_type,
                      bool copy) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}

at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}

}  // namespace torch_lazy_tensors
