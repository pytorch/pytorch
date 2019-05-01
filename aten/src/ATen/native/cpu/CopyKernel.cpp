#include <ATen/native/cpu/CopyKernel.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Copy.h>

namespace at {
namespace native {
namespace {

template <typename self_T>
void copy_kernel_cast_t_impl(Tensor& self, const Tensor& src) {
  auto builder = TensorIterator::Builder();
  builder.add_output(self);
  builder.add_input(src);
  builder.dont_resize_outputs();
  builder.dont_compute_common_dtype();
  auto iter = builder.build();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      src.scalar_type(),
      "copy_kernel_cast",
      [&] {
        at::native::unary_kernel(*iter, [=](scalar_t a) -> self_T {
          return static_cast<self_T>(
              static_cast<at::native::inter_copy_type_t<self_T>>(a));
        });
      });
}

static void copy_kernel_cast_impl(Tensor& self, const Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool,
      self.scalar_type(), "copy_kernel_cast", [&]() { copy_kernel_cast_t_impl<scalar_t>(self, src); });
}

static void copy_kernel_same_type_impl(Tensor& self, const Tensor& src) {
  auto builder = TensorIterator::Builder();
  builder.add_output(self);
  builder.add_input(src);
  builder.dont_resize_outputs();
  auto iter = builder.build();

  if (self.scalar_type() == at::ScalarType::Half) {
    unary_kernel(*iter, [=](at::Half a) -> at::Half { return a; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Bool, self.scalar_type(), "copy_kernel_same_type", [&] {
          unary_kernel_vec(
              *iter,
              [=](scalar_t a) -> scalar_t { return a; },
              [=](Vec256<scalar_t> a) { return a; });
        });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_kernel_same_type, &copy_kernel_same_type_impl);
REGISTER_DISPATCH(copy_kernel_cast, &copy_kernel_cast_impl);

} // namespace native
} // namespace at
