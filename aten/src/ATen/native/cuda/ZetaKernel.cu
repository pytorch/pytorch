#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>

namespace at { namespace native {
namespace {

void zeta_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
      return zeta<scalar_t, /*is_cuda=*/true>(x, q);
    });
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(zeta_stub, &zeta_kernel_cuda);

}} // namespace at::native
