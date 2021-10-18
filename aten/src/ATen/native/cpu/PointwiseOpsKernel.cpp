// Ternary and higher-order pointwise operations
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/Scalar.h>
#include <ATen/native/cpu/zmath.h>

namespace at {
namespace native {
namespace {

static void addcmul_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  ScalarType dtype = iter.dtype(0);
  if (iter.dtype() == kBFloat16) {
    float float_val = value.to<float>();
    auto float_vec = Vectorized<float>(float_val);
    cpu_kernel_vec(
        iter,
        [=](BFloat16 self_val, BFloat16 t1_val, BFloat16 t2_val) -> BFloat16 {
          return float(self_val) + float_val * float(t1_val) * float(t2_val);
        },
        [=](Vectorized<BFloat16> self_vec,
          Vectorized<BFloat16> t1_vec,
          Vectorized<BFloat16> t2_vec) {
          Vectorized<float> self_vec0, self_vec1;
          std::tie(self_vec0, self_vec1) = convert_bfloat16_float(self_vec);
          Vectorized<float> t1_vec0, t1_vec1, t2_vec0, t2_vec1;
          std::tie(t1_vec0, t1_vec1) = convert_bfloat16_float(t1_vec);
          std::tie(t2_vec0, t2_vec1) = convert_bfloat16_float(t2_vec);
          self_vec0 = self_vec0 + float_vec * t1_vec0 * t2_vec0;
          self_vec1 = self_vec1 + float_vec * t1_vec1 * t2_vec1;
          return convert_float_bfloat16(self_vec0, self_vec1);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcmul_cpu_out", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val * t2_val;
          },
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec * t2_vec;
          });
    });
  }
}

static void addcdiv_cpu_kernel(TensorIteratorBase& iter, const Scalar& value) {
  ScalarType dtype = iter.dtype(0);
  if (dtype == kBFloat16) {
    float float_val = value.to<float>();
    auto float_vec = Vectorized<float>(float_val);
    cpu_kernel_vec(
        iter,
        [=](BFloat16 self_val, BFloat16 t1_val, BFloat16 t2_val) -> BFloat16 {
          return float(self_val) + float_val * float(t1_val) / float(t2_val);
        },
        [=](Vectorized<BFloat16> self_vec,
            Vectorized<BFloat16> t1_vec,
            Vectorized<BFloat16> t2_vec) {
            Vectorized<float> self_vec0, self_vec1;
          std::tie(self_vec0, self_vec1) = convert_bfloat16_float(self_vec);
          Vectorized<float> t1_vec0, t1_vec1, t2_vec0, t2_vec1;
          std::tie(t1_vec0, t1_vec1) = convert_bfloat16_float(t1_vec);
          std::tie(t2_vec0, t2_vec1) = convert_bfloat16_float(t2_vec);
          self_vec0 = self_vec0 + float_vec * t1_vec0 / t2_vec0;
          self_vec1 = self_vec1 + float_vec * t1_vec1 / t2_vec1;
          return convert_float_bfloat16(self_vec0, self_vec1);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcdiv_cpu_out", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto scalar_vec = Vectorized<scalar_t>(scalar_val);
      cpu_kernel_vec(
          iter,
          [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
            return self_val + scalar_val * t1_val / t2_val;
          },
          [=](Vectorized<scalar_t> self_vec,
              Vectorized<scalar_t> t1_vec,
              Vectorized<scalar_t> t2_vec) {
            return self_vec + scalar_vec * t1_vec / t2_vec;
          });
    });
  }
}

void l1_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "l1_backward_cpu_complex", [&]() {
      using Vec = Vectorized<scalar_t>;
      const auto norm_val = norm.to<scalar_t>();
      const auto norm_val_vec = Vec(norm_val);
      cpu_kernel_vec(iter,
        [norm_val](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
          return sgn_impl(input - target) * norm_val * grad_output;
        },
        [norm_val_vec](Vec input, Vec target, Vec grad_output) -> Vec {
          return (input - target).sgn() * norm_val_vec * grad_output;
        });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "l1_backward_cpu_real", [&]() {
      using Vec = Vectorized<scalar_t>;
      const auto norm_val = norm.to<scalar_t>();
      const auto norm_val_vec = Vec(norm_val);
      const auto zero_vec = Vec(static_cast<scalar_t>(0));
      cpu_kernel_vec(iter,
        [norm_val](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
          const auto x = input - target;
          return static_cast<scalar_t>(((0 < x) - (x < 0))) * norm_val * grad_output;
        },
        [norm_val_vec, zero_vec](Vec input, Vec target, Vec grad_output) -> Vec {
          const auto x = input - target;
          const auto y = norm_val_vec * grad_output;
          return Vec::blendv(y.neg(), y, x > zero_vec);
        });
    });
  }
}

static void smooth_l1_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "smooth_l1_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto norm_val = norm.to<scalar_t>();
    scalar_t beta_val(beta);
    auto norm_val_vec = Vec(norm_val);
    auto beta_val_vec = Vec(beta_val);
    const auto neg_1_vec = Vec(-1);
    const auto zero_vec = Vec(0);
    const auto pos_1_vec = Vec(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x <= -beta)
          return -norm_val * grad_output;
        else if (x >= beta)
          return norm_val * grad_output;
        else
          return norm_val * x * grad_output / beta;
      },
      [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vec input, Vec target, Vec grad_output) -> Vec {
        // using two blendv calls to simulate the 3 cases
        // 1        if  x >= beta
        // -1       if x <= -beta
        // x / beta if |x| < beta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vec::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vec::blendv(
            x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

static void huber_backward_cpu_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "huber_backward_cpu_out", [&] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t delta_val(delta);
    auto norm_val_vec = Vectorized<scalar_t>(norm_val);
    auto delta_val_vec = Vectorized<scalar_t>(delta_val);
    const auto neg_1_vec = Vectorized<scalar_t>(-1);
    const auto zero_vec = Vectorized<scalar_t>(0);
    const auto pos_1_vec = Vectorized<scalar_t>(1);
    cpu_kernel_vec(iter,
      [=](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x <= -delta) {
          return -norm_val * grad_output * delta;
        } else if (x >= delta) {
          return norm_val * grad_output * delta;
        } else {
          return norm_val * x * grad_output;
        }
      },
      [norm_val_vec, delta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
         Vectorized<scalar_t> input, Vectorized<scalar_t> target, Vectorized<scalar_t> grad_output) -> Vectorized<scalar_t> {
        // using two blendv calls to simulate the 3 cases
        // delta     if  x >= delta
        // -delta    if x <= -delta
        // x        if |x| < delta
        const auto x = input - target;
        const auto pos_or_neg_1_vec = Vectorized<scalar_t>::blendv(
            neg_1_vec, pos_1_vec, x > zero_vec);
        const auto x_abs = x.abs();
        const auto output = Vectorized<scalar_t>::blendv(
            x, pos_or_neg_1_vec * delta_val_vec, x_abs >= delta_val_vec);
        return norm_val_vec * output * grad_output;
      }
    );
  });
}

static void mse_backward_cpu_kernel(TensorIterator& iter, const Scalar& value) {
  ScalarType dtype = iter.dtype(0);
  AT_DISPATCH_ALL_TYPES(dtype, "mse_backward_cpu_out", [&] {
    scalar_t scalar_val = value.to<scalar_t>();
    auto scalar_vec = Vectorized<scalar_t>(scalar_val);
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val, scalar_t t1_val, scalar_t t2_val) -> scalar_t {
          return scalar_val * (self_val - t1_val) * t2_val;
        },
        [=](Vectorized<scalar_t> self_vec,
            Vectorized<scalar_t> t1_vec,
            Vectorized<scalar_t> t2_vec) {
          return scalar_vec * (self_vec - t1_vec) *  t2_vec;
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addcmul_stub, &addcmul_cpu_kernel);
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cpu_kernel);
REGISTER_DISPATCH(l1_backward_stub, &l1_backward_cpu_kernel);
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cpu_kernel);
REGISTER_DISPATCH(huber_backward_stub, &huber_backward_cpu_kernel);
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_cpu_kernel);

} // namespace native
} // namespace at
