#include <ATen/native/Activation.h>

#include <ATen/ATen.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {
namespace {

static void threshold_kernel(TensorIterator& iter, Scalar threshold_scalar, Scalar value_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "threshold", [&] {
    using Vec = Vec256<scalar_t>;
    scalar_t threshold = threshold_scalar.to<scalar_t>();
    scalar_t value = value_scalar.to<scalar_t>();
    binary_kernel_vec(
      iter,
      [&](scalar_t x, scalar_t other) -> scalar_t {
        return x <= threshold ? value : other;
      },
      [&](Vec x, Vec other) -> Vec {
        return Vec::blendv(other, Vec(value), x <= Vec(threshold));
      });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);

}} // namespace at::native
