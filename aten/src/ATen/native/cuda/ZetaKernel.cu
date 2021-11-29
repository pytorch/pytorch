#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/jit_utils.h>

namespace at { namespace native {
namespace {

/*
 * This function is derived from the implementation of the zeta function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
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
