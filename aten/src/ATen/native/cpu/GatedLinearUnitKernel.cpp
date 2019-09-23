#include <ATen/native/GatedLinearUnit.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {
namespace {

using namespace vec256;

void glu_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&]() {
    cpu_kernel_vec(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
      return a * (1 / (1 + std::exp(-b)));
    },
    [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
      b = Vec256<scalar_t>((scalar_t)(0)) - b;
      b = b.exp();
      b = Vec256<scalar_t>((scalar_t)(1)) + b;
      b = b.reciprocal();
      a = a * b;
      return a;
    });
  });
}

} // namespace

REGISTER_DISPATCH(glu_stub, &glu_kernel);

} // namespace native
} // namespace at
