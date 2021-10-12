#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <ATen/native/Activation.h>

#include <cmath>
#include <functional>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif // AT_MKL_ENABLED()

namespace at {
namespace native {

namespace {

template <typename scalar_t>
inline void _vec_log_sigmoid(Tensor& output, Tensor& buffer, const Tensor& input) {
  using Vec = Vectorized<scalar_t>;
  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
  scalar_t* input_data = input.data_ptr<scalar_t>();
  parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
    int64_t size = end - begin;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec data_vec = Vec::loadu(input_data + begin+ d);
      Vec max_vec = vec::maximum(data_vec.neg(), Vec(scalar_t(0)));
      Vec buffer_vec =  max_vec.neg().exp() + (data_vec.neg() - max_vec).exp();
      Vec output_vec = (max_vec + buffer_vec.log()).neg();
      buffer_vec.store(buffer_data + begin + d);
      output_vec.store(output_data + begin + d);
    }
    if (size - d > 0) {
      Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
      Vec max_vec = vec::maximum(data_vec.neg(), Vec(scalar_t(0)));
      Vec buffer_vec =  max_vec.neg().exp() + (data_vec.neg() - max_vec).exp();
      Vec output_vec = (max_vec + buffer_vec.log()).neg();
      buffer_vec.store(buffer_data + begin + d, size - d);
      output_vec.store(output_data + begin + d, size - d);
    }
  });
}

static void log_sigmoid_cpu_kernel(Tensor& output, Tensor& buffer, const Tensor& input) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&] {
    _vec_log_sigmoid<scalar_t>(output, buffer, input);
  });
}

static void log_sigmoid_backward_cpu_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto zero_val = scalar_t(0);
    auto zero_vec = Vec(zero_val);
    auto one_val = scalar_t(1);
    auto one_vec = Vec(one_val);
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        auto max_deriv_val = zero_val;
        auto sign_val = -one_val;
        if (a < zero_val) {
          max_deriv_val = -one_val;
          sign_val = one_val;
        }
        return (-max_deriv_val - sign_val * ((b - one_val) / b)) * c;
      },
      [=](Vec a, Vec b, Vec c) -> Vec {
        auto mask = a < zero_vec;
        auto max_deriv_vec = Vec::blendv(zero_vec, one_vec.neg(), mask);
        auto sign_vec = Vec::blendv(one_vec.neg(), one_vec, mask);
        return (max_deriv_vec + sign_vec * ((b - one_vec) / b)).neg() * c;
      });
  });
}

static void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold_scalar,
    const Scalar& value_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "threshold_cpu", [&] {
    using Vec = Vectorized<scalar_t>;
    scalar_t threshold = threshold_scalar.to<scalar_t>();
    Vec threshold_v = Vec(threshold);
    scalar_t value = value_scalar.to<scalar_t>();
    Vec value_v = Vec(value);
    cpu_kernel_vec(
        iter,
        [&](scalar_t x, scalar_t other) -> scalar_t {
          return x <= threshold ? value : other;
        },
        [&](Vec x, Vec other) -> Vec {
          return Vec::blendv(other, value_v, x <= threshold_v);
        });
  });
}

void elu_kernel(TensorIteratorBase& it, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    const Vec negcoef_vec(negcoef);
    const Vec negiptcoef_vec(negiptcoef);
    const Vec poscoef_vec(poscoef);
    const Vec one_vec(static_cast<scalar_t>(1));
    const Vec zero_vec(static_cast<scalar_t>(0));
    cpu_kernel_vec(
        it,
        [negcoef, negiptcoef, poscoef](scalar_t a) -> scalar_t {
          return a <= scalar_t(0) ? (std::exp(a * negiptcoef) - scalar_t(1)) * negcoef : a * poscoef;
        },
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &one_vec, &zero_vec](Vec a) -> Vec {
          auto cmp = (a > zero_vec);
          if (!cmp.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
            return a * poscoef_vec;
          } else {
            return Vec::blendv(((a * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a * poscoef_vec, cmp);
          }
        });
  });
}

void elu_backward_kernel(TensorIteratorBase& it, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, bool is_result) {
  AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    const Vec negcoef_vec(negcoef);
    const Vec negiptcoef_vec(negiptcoef);
    const Vec poscoef_vec(poscoef);
    const Vec zero_vec(static_cast<scalar_t>(0));
    cpu_kernel_vec(
        it,
        [negcoef, negiptcoef, poscoef, is_result](scalar_t a, scalar_t b) -> scalar_t {
          if (is_result) {
            return b <= scalar_t(0) ? a * negiptcoef * (b + negcoef) : a * poscoef;
          } else {
            return b <= scalar_t(0) ? a * negiptcoef * negcoef * std::exp(b * negiptcoef): a * poscoef;
          }
        },
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &zero_vec, is_result](Vec a, Vec b) -> Vec {
          auto cmp = (b > zero_vec);
          if (is_result) {
            if (!cmp.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
              return a * poscoef_vec;
            } else {
              return Vec::blendv(a * negiptcoef_vec * (b + negcoef_vec), a * poscoef_vec, cmp);
            }
          } else {
            return Vec::blendv(a * negiptcoef_vec * negcoef_vec * (b * negiptcoef_vec).exp(), a * poscoef_vec, cmp);
          }
        }
    );
  });
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.
void GeluKernelImpl(TensorIteratorBase& it) {
  auto grain_size = at::internal::GRAIN_SIZE;
  // Numbers based on benchmarking.
  // Benchmark: benchmarks/operator_benchmarks/pt/gelu_test.py
#ifdef C10_MOBILE
  // Benchmarked on S8 US phone.
  // Internal benchmarking that converts operator benchmark into
  // a torchscript module and run that on mobile.
  // Same benchmark as server side.
  constexpr int64_t GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{6144};
#else
  // Benchmarked on i9 8 core 16 thread machine.
  // 1 thread: cd benchmark/operator_benchmarks;
  //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  // 2 threads: cd benchmark/operator_benchmarks;
  //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
  constexpr int64_t GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{16384};
#endif
  if (it.numel() > GELU_MIN_ELEMENTS_FOR_MULTI_THREADING) {
    grain_size = it.numel() / at::get_num_threads();
  }
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, it.dtype(), "GeluKernelImpl", [&]() {
    using Vec = vec::Vectorized<scalar_t>;
    const Vec kAlphaVec(scalar_t(M_SQRT1_2));
    const Vec kOneVec(scalar_t(1));
    const Vec kPointFiveVec(scalar_t(0.5));
    cpu_kernel_vec(
        it,
        [](scalar_t x) {
          const scalar_t kAlpha = scalar_t(M_SQRT1_2);
          return x * scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
        },
        [&](Vec x_vec) {
          return x_vec * kPointFiveVec *
              (kOneVec + (x_vec * kAlphaVec).erf());
        },
        grain_size);
  });
}

void GeluBackwardKernelImpl(TensorIteratorBase& it) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, it.dtype(), "GeluBackwardKernelImpl", [&]() {
    using Vec = vec::Vectorized<scalar_t>;
    const Vec kAlphaVec(scalar_t(M_SQRT1_2));
    const Vec kBetaVec(scalar_t(M_2_SQRTPI * M_SQRT1_2 * 0.5));
    const Vec kOneVec(scalar_t(1));
    const Vec kPointFiveVec(scalar_t(0.5));
    const Vec kMinusPointFiveVec(scalar_t(-0.5));
    cpu_kernel_vec(
        it,
        [](scalar_t dy, scalar_t x) {
          const scalar_t kAlpha = scalar_t(M_SQRT1_2);
          const scalar_t kBeta = M_2_SQRTPI * M_SQRT1_2 * scalar_t(0.5);
          const scalar_t cdf =
              scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
          const scalar_t pdf = kBeta * std::exp(x * x * scalar_t(-0.5));
          return dy * (cdf + x * pdf);
        },
        [&](Vec dy_vec, Vec x_vec) {
          const Vec cdf_vec =
              kPointFiveVec * (kOneVec + (x_vec * kAlphaVec).erf());
          const Vec pdf_vec =
              kBetaVec * (x_vec * x_vec * kMinusPointFiveVec).exp();
          return dy_vec * (cdf_vec + x_vec * pdf_vec);
        });
  });
}

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_cpu", [&] {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t six(6.0f);
    using Vec = vec::Vectorized<scalar_t>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    cpu_kernel_vec(
        iter,
        [&](scalar_t self_val) {
          return std::min(std::max(self_val + three, zero), six) / six;
        },
        [&](Vec self_val) {
          return vec::minimum(
            vec::maximum(self_val + kThreeVec, kZeroVec),
            kSixVec
          ) / kSixVec;
        });
  });
}

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_backward", [&] {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t neg_three(-3.0f);
    const scalar_t one_sixth(1.0f / 6.0f);
    using Vec = Vectorized<scalar_t>;
    Vec kZeroVec(0.0f);
    Vec kOneSixthVec(1.0f / 6.0f);
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val > neg_three && self_val < three)
            ? grad_val * one_sixth
            : zero;
        },
        [=](Vec grad_val, Vec self_val) {
          Vec gradNonZeroMask = (self_val > neg_three) & (self_val < three);
          return Vec::blendv(kZeroVec, grad_val * kOneSixthVec, gradNonZeroMask);
        });
  });
}

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : self_val;
        },
        [=](Vectorized<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & self_val;
        });
  });
}

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softshrink_cpu", [&]() {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
      return a > lambd_val ? a - lambd_val : (a < -lambd_val ? a + lambd_val : scalar_t(0));
    });
  });
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "shrink_backward_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : grad_val;
        },
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & grad_val;
        });
  });
}

void hardtanh_backward_kernel(TensorIterator& iter, const Scalar& min, const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&] {
    auto min_val = min.to<scalar_t>();
    auto max_val = max.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val <= min_val || self_val >= max_val) ? scalar_t(0) : grad_val;
        },
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) {
          return ((self_val > min_val) & (self_val < max_val)) & grad_val;
        });
  });
}

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_cpu", [&]() {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t six(6.0f);
    using Vec = vec::Vectorized<scalar_t>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    cpu_kernel_vec(
      iter,
      [&](scalar_t x) {
        return x * std::min(std::max(x + three, zero), six) / six;
      },
      [&](Vec x_vec) {
        return x_vec * vec::minimum(
          vec::maximum(x_vec + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
      }
    );
  });
}

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_backward_cpu", [&]() {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t neg_three(-3.0f);
    const scalar_t one_half(0.5f);
    using Vec = vec::Vectorized<scalar_t>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kNegThreeVec(neg_three);
    const Vec kOneHalfVec(one_half);
    cpu_kernel_vec(
      iter,
      [&](scalar_t grad_val, scalar_t self_val) {
        if (self_val < neg_three) {
          return zero;
        } else if (self_val <= three) {
          return grad_val * ((self_val / three) + one_half);
        } else {
          return grad_val;
        }
      },
      [&](Vec grad_val, Vec self_val) {
        return Vec::blendv(
          Vec::blendv(
            grad_val * ((self_val / kThreeVec) + kOneHalfVec),
            grad_val,
            self_val >= kThreeVec
          ),
          kZeroVec,
          self_val < kNegThreeVec
        );
      }
    );
  });
}

static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  if (iter.common_dtype() == kBFloat16) {
    auto zero_vec = Vectorized<float>((float)(0));
    auto one_vec = Vectorized<float>((float)(1));
    float negval = negval_.to<float>();
    Vectorized<float> negval_v = Vectorized<float>(negval);
    cpu_kernel_vec(
        iter,
        [&](BFloat16 a) -> BFloat16 {
          return float(a) > float(0) ? float(a) : float(a) * negval;
        },
        [&](Vectorized<BFloat16> a) -> Vectorized<BFloat16> {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          auto res0 = a0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
          auto res1 = a1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
          return convert_float_bfloat16(res0, res1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&] {
      using Vec = Vectorized<scalar_t>;
      auto zero_vec = Vec((scalar_t)(0));
      auto one_vec = Vec((scalar_t)(1));
      scalar_t negval = negval_.to<scalar_t>();
      Vec negval_v = Vec(negval);
      cpu_kernel_vec(
          iter,
          [&](scalar_t a) -> scalar_t {
            return a > scalar_t(0) ? a : a * negval;
          },
          [&](Vec a) -> Vec {
            auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
            return a * r;
          });
    });
  }
}

static void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  if (iter.common_dtype() == kBFloat16) {
    auto zero_vec = Vectorized<float>((float)(0));
    auto one_vec = Vectorized<float>((float)(1));
    float negval = negval_.to<float>();
    Vectorized<float> negval_v = Vectorized<float>(negval);
    cpu_kernel_vec(
      iter,
      [&](BFloat16 a, BFloat16 b) -> BFloat16 {
        return float(a) > float(0) ? float(b) : float(b) * negval;
      },
      [&](Vectorized<BFloat16> a, Vectorized<BFloat16> b) -> Vectorized<BFloat16> {
        Vectorized<float> a0, a1, b0, b1;
        std::tie(a0, a1) = convert_bfloat16_float(a);
        std::tie(b0, b1) = convert_bfloat16_float(b);
        auto res0 = b0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
        auto res1 = b1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
        return convert_float_bfloat16(res0, res1);
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&] {
      using Vec = Vectorized<scalar_t>;
      auto zero_vec = Vec((scalar_t)(0));
      auto one_vec = Vec((scalar_t)(1));
      scalar_t negval = negval_.to<scalar_t>();
      Vec negval_v = Vec(negval);
      cpu_kernel_vec(
          iter,
          [&](scalar_t a, scalar_t b) -> scalar_t {
            return a > scalar_t(0) ? b : b * negval;
          },
          [&](Vec a, Vec b) -> Vec {
            auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
            return b * r;
          });
    });
  }
}

void softplus_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    cpu_kernel_vec(
        iter,
        [beta, threshold](scalar_t a) -> scalar_t {
          return (a * beta) > threshold ? a
            : static_cast<scalar_t>(std::log1p(std::exp(a * beta))) / beta;
        },
        [beta_vec, threshold_vec](Vec a) -> Vec {
          return Vec::blendv((a * beta_vec).exp().log1p() / beta_vec, a, (a * beta_vec) > threshold_vec);
        }
    );
  });
}

void softplus_backward_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    const Vec one_vec(static_cast<scalar_t>(1.0));
    cpu_kernel_vec(
        iter,
        [beta, threshold](scalar_t a, scalar_t b) -> scalar_t {
          scalar_t z = std::exp(b * beta);
          return (b * beta) > threshold ? a : a * z / (z + scalar_t(1.));
        },
        [beta_vec, one_vec, threshold_vec](Vec a, Vec b) -> Vec {
          const Vec z = (b * beta_vec).exp();
          return Vec::blendv(a * z / (z + one_vec), a, (b * beta_vec) > threshold_vec);
        }
    );
  });
}

void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&] {
    using Vec = Vectorized<scalar_t>;
    const scalar_t one_val(1);
    const Vec one_vec(one_val);
    cpu_kernel_vec(
      iter,
      [one_val](scalar_t a, scalar_t b) -> scalar_t {
        return a * (one_val / (one_val + std::exp(-b)));
      },
      [one_vec](Vec a, Vec b) -> Vec {
        return a * (one_vec / (one_vec + b.neg().exp()));
      }
    );
  });
}

void glu_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_backward_cpu", [&] {
    using Vec = Vectorized<scalar_t>;
    const scalar_t one_val(1);
    const Vec one_vec(one_val);
    cpu_kernel_vec(
      iter,
      [one_val](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return (one_val - a) * a * b * c;
      },
      [one_vec](Vec a, Vec b, Vec c) -> Vec {
        return (one_vec - a) * a * b * c;
      }
    );
  });
}

void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      kBFloat16, iter.dtype(), "silu_cpu", [&]() {
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        cpu_kernel_vec(
            iter,
            [](scalar_t x) {
              return x / (scalar_t(1) + std::exp(-x));
            },
            [kOneVec](Vectorized<scalar_t> x_vec) {
              return x_vec / (kOneVec + x_vec.neg().exp());
            });
      });
}

void silu_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      kBFloat16, iter.dtype(), "silu_backward_cpu", [&]() {
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        cpu_kernel_vec(
            iter,
            [](scalar_t dy, scalar_t x) {
              const scalar_t sigmoid =
                  scalar_t(1) / (scalar_t(1) + std::exp(-x));
              return dy * sigmoid * (scalar_t(1) + x * (scalar_t(1) - sigmoid));
            },
            [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
              const Vectorized<scalar_t> sigmoid =
                  kOneVec / (kOneVec + x_vec.neg().exp());
              return dy_vec * sigmoid * (kOneVec + x_vec * (kOneVec - sigmoid));
            });
      });
}

void mish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_cpu", [&]() {
        using Vec = Vectorized<scalar_t>;
        cpu_kernel_vec(
            iter,
            [](scalar_t x) -> scalar_t{
              return static_cast<scalar_t>(x * std::tanh(std::log1p(std::exp(x))));
            },
            [](Vec x_vec) -> Vec {
              return x_vec * x_vec.exp().log1p().tanh();
            });
      });
}

void mish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_backward_cpu", [&]() {
        using Vec = Vectorized<scalar_t>;
        const Vec kOneVec(scalar_t(1));
        cpu_kernel_vec(
            iter,
            [](scalar_t dy, scalar_t x) -> scalar_t {
              const scalar_t sigmoid =
                  scalar_t(1) / (scalar_t(1) + std::exp(-x));
              const scalar_t tanh_softplus = std::tanh(std::log1p(std::exp(x)));
              return dy * (tanh_softplus + x * sigmoid * (scalar_t(1) - tanh_softplus * tanh_softplus));
            },
            [kOneVec](Vec dy_vec, Vec x_vec) -> Vec {
              const Vec sigmoid = kOneVec / (kOneVec + x_vec.neg().exp());
              const Vec tanh_softplus = x_vec.exp().log1p().tanh();
              return dy_vec * (tanh_softplus + x_vec * sigmoid * (kOneVec - tanh_softplus * tanh_softplus));
            });
      });
}

} // namespace

REGISTER_DISPATCH(log_sigmoid_cpu_stub, &log_sigmoid_cpu_kernel);
REGISTER_DISPATCH(log_sigmoid_backward_stub, &log_sigmoid_backward_cpu_kernel);
REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
REGISTER_DISPATCH(elu_stub, &elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel);
REGISTER_DISPATCH(GeluKernel, &GeluKernelImpl);
REGISTER_DISPATCH(GeluBackwardKernel, &GeluBackwardKernelImpl);
REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel);
REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel);
REGISTER_DISPATCH(hardswish_stub, &hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, &hardswish_backward_kernel);
REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel);
REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel);
REGISTER_DISPATCH(softplus_stub, &softplus_kernel);
REGISTER_DISPATCH(softplus_backward_stub, &softplus_backward_kernel);
REGISTER_DISPATCH(glu_stub, &glu_kernel);
REGISTER_DISPATCH(glu_backward_stub, &glu_backward_kernel);
REGISTER_DISPATCH(silu_stub, &silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, &silu_backward_kernel);
REGISTER_DISPATCH(mish_stub, &mish_kernel);
REGISTER_DISPATCH(mish_backward_stub, &mish_backward_kernel);

} // namespace native
} // namespace at
