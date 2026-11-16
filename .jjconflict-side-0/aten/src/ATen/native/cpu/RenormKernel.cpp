#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Normalization.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/cpu/vec/vec.h>

#include <ATen/Dispatch.h>

namespace at::native {
namespace {

void renorm_scale_factor_impl(TensorIteratorBase& iter, double maxnorm) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "renorm_scale_factor_cpu", [&] {
    using vec_t = at::vec::Vectorized<scalar_t>;
    const auto maxnorm_s = static_cast<scalar_t>(maxnorm);
    const auto maxnorm_v = vec_t(maxnorm_s);
    const auto eps_v = vec_t(static_cast<scalar_t>(1e-7));
    const auto one_v = vec_t(1.0);
    cpu_kernel_vec(
      iter,
      [maxnorm_s](scalar_t norm) -> scalar_t {
        const auto eps = static_cast<scalar_t>(1e-7);
        return (norm > maxnorm_s) ?
            maxnorm_s / (norm + eps) : static_cast<scalar_t>(1.0);
      },
      [maxnorm_v, eps_v, one_v](vec_t norm) -> vec_t {
        auto fct = maxnorm_v / (norm + eps_v);
        return vec_t::blendv(one_v, fct, norm > maxnorm_v);
      });
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(renorm_scale_factor_stub, &renorm_scale_factor_impl)

}  // namespace at::native
