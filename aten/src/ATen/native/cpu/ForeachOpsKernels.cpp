#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ForeachOps.h>

namespace at { namespace native {
namespace {

static std::vector<Tensor> foreach_add_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  std::vector<Tensor> result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar_kernel_cpu", [&]() {
    for (int i = 0; i < tensors.size(); i++) {
      auto temp = tensors[i].add(scalar.to<scalar_t>());
      result.push_back(temp);
    }
  });
  return result;
}

} // anonymous namespace

REGISTER_DISPATCH(foreach_tensor_add_scalar_stub, &foreach_add_scalar_kernel_cpu);

}} // namespace at::native
