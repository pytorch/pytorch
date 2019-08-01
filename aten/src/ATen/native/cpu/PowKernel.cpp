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
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      cpu_kernel_vec(iter,
        [=](scalar_t self, scalar_t exp) -> scalar_t {
          return std::pow(self, exp);
        },
        [&](Vec self, Vec exp) -> Vec {
          return self.pow(exp);
        }
      );
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t self, scalar_t exp) -> scalar_t {
          return std::pow(self, exp);
        }
      );
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      auto exp = exp_scalar.to<double>();
      if (exp == 0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return std::sqrt((long double)self); },
          [](Vec self) -> Vec { return self.sqrt(); }
        );
      } else if (exp == 2) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return self * self; },
          [](Vec self) -> Vec { return self * self; }
        );
      } else if (exp == 3) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return self * self * self; },
          [](Vec self) -> Vec { return self * self * self; }
        );
      } else if (exp == -0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return 1.0 / std::sqrt((long double)self); },
          [](Vec self) -> Vec { return self.sqrt().reciprocal(); }
        );
      } else if (exp == -1) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return 1.0 / self; },
          [](Vec self) -> Vec { return self.reciprocal(); }
        );
      } else if (exp == -2) {
        cpu_kernel_vec(iter,
          [](scalar_t self) -> scalar_t { return 1.0 / (self * self); },
          [](Vec self) -> Vec { return (self * self).reciprocal(); }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t self) -> scalar_t { return std::pow((long double)self, exp); },
          [=](Vec self) -> Vec { return self.pow(exp); }
        );
      }
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      auto exp = exp_scalar.to<double>(); // because of correctness
      if (exp == 0.5) {
        cpu_kernel(iter,
          [](scalar_t self) -> scalar_t { return std::sqrt(self); }
        );
      // } else if (exp == 2) {
      //   cpu_kernel(iter,
      //     [](scalar_t self) -> scalar_t { return self * self; }
      //   );
      // } else if (exp == 3) {
      //   cpu_kernel(iter,
      //     [](scalar_t self) -> scalar_t { return self * self * self; }
      //   );
      } else if (exp == -0.5) {
        cpu_kernel(iter,
          [](scalar_t self) -> scalar_t { return 1.0 / std::sqrt(self); }
        );
      } else if (exp == -1) {
        cpu_kernel(iter,
          [](scalar_t self) -> scalar_t { return 1.0 / self; }
        );
      // } else if (exp == -2) {
      //   cpu_kernel(iter,
      //     [](scalar_t self) -> scalar_t { return 1.0 / (self * self); }
      //   );
      } else {
        cpu_kernel(iter,
          [=](scalar_t self) -> scalar_t { return std::pow((long double)self, exp); } // TODO: long double perf test
        );
      }
    });
  }
}

void pow_scalar_tensor_kernel(TensorIterator& iter, Scalar self_scalar) {
  if (self_scalar.isFloatingPoint()) {
    const auto self = self_scalar.to<double>();
    AT_DISPATCH_ALL_TYPES(iter.input(0).scalar_type(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t exp) -> double { return std::pow((long double)self, exp); }
      );
    });
  } else {
    const auto self = self_scalar.to<long>();
    AT_DISPATCH_ALL_TYPES(iter.input(0).scalar_type(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t exp) -> long {
          return (long)std::pow((long double)self, (long double)exp);
        }
      );
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);
REGISTER_DISPATCH(pow_scalar_tensor_stub, &pow_scalar_tensor_kernel);

}} // namespace at::native
