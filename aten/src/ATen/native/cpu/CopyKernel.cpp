#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <iostream>

namespace at {
namespace native {
namespace {

template <typename self_T>
void copy_kernel_cast(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16,
      iter.dtype(1),
      "copy_kernel_cast",
      [&] {
        cpu_kernel(iter, [=](scalar_t a) -> self_T {
          if (iter.dtype(1) == ScalarType::BFloat16) {
            return (self_T)a;
          }
          //std::cout << "self " << typeid(self_T).name() << std::endl;
          //std::cout << "scal " << typeid(scalar_t).name() << std::endl;
          //std::cout << "a: " << (float)a << std::endl;
          self_T res = static_cast<self_T>(
              static_cast<at::native::inter_copy_type_t<self_T>>(a));
          //std::cout << "res: " << (float)res << std::endl;
          return res;
        });
      });
}

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  ScalarType dtype = iter.dtype(0);
  if (dtype == iter.dtype(1)) {
    if (dtype == ScalarType::Half) {
      cpu_kernel(iter, [=](at::Half a) -> at::Half { return a; });
    } else if (dtype == ScalarType::BFloat16) {
      cpu_kernel(iter, [=](at::BFloat16 a) -> at::BFloat16 { return a; });
    } else if (isQIntType(dtype)) {
      AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t {return a; });
      });
    } else {
      AT_DISPATCH_ALL_TYPES_AND(
          ScalarType::Bool, dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](Vec256<scalar_t> a) { return a; });
          });
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      copy_kernel_cast<scalar_t>(iter);
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
