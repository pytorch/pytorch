#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace {

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "pow", [&]() {
    using Vec = Vec256<scalar_t>;
    cpu_kernel(iter,
      [=](scalar_t self, scalar_t exp) -> scalar_t {
        return std::pow(self, exp);
      }
    );

    // TODO: AT_DISPATCH_FLOATING_TYPES ?
    // cpu_kernel_vec(iter,
    //   [=](scalar_t self, scalar_t exp) -> scalar_t {
    //     return std::pow(self, exp);
    //   },
    //   [&](Vec self, Vec exp) -> Vec {
    //     return self.pow(exp);
    //   }
    // );
  });
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "pow", [&]() {
    auto exp = exp_scalar.to<scalar_t>();
    if (exp == 0.5) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return std::sqrt(self); }
      );
    } else if (exp == 2) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return self * self; }
      );
    } else if (exp == 3) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return self * self * self; }
      );
    } else if (exp == -0.5) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return 1.0 / std::sqrt(self); }
      );
    } else if (exp == -1) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return 1.0 / self; }
      );
    } else if (exp == -2) {
      cpu_kernel(iter,
        [](scalar_t self) -> scalar_t { return 1.0 / (self * self); }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self) -> scalar_t { return std::pow(self, exp); }
      );
    }
  });
}

void pow_scalar_tensor_kernel(TensorIterator& iter, Scalar self_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "pow", [&]() {
    auto self = self_scalar.to<scalar_t>();
    cpu_kernel(iter,
      [=](scalar_t exp) -> scalar_t { return std::pow(self, exp); }
    );
  });
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);
REGISTER_DISPATCH(pow_scalar_tensor_stub, &pow_scalar_tensor_kernel);

}} // namespace at::native
