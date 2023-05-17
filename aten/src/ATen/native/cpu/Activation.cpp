#define TORCH_ASSERT_NO_OPERATORS
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <ATen/native/Activation.h>


#include <cmath>
#include <functional>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>

#include <c10/core/Scalar.h>

namespace at::native {

namespace {

static void log_sigmoid_cpu_kernel(TensorBase &output, TensorBase &buffer, const TensorBase &input) {
  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    scalar_t* output_data = output.data_ptr<scalar_t>();
    scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
    scalar_t* input_data = input.data_ptr<scalar_t>();
    parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
      int64_t size = end - begin;
      int64_t d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(input_data + begin+ d);
        Vectorized<float> data_vec0, data_vec1;
        std::tie(data_vec0, data_vec1) = convert_to_float<scalar_t>(data_vec);
        Vectorized<float> min_vec = minimum(data_vec0, Vectorized<float>(float(0)));
        Vectorized<float> buffer_vec0 = data_vec0.abs().neg().exp();
        Vectorized<float> output_vec0 = min_vec - buffer_vec0.log1p();
        min_vec = minimum(data_vec1, Vectorized<float>(float(0)));
        Vectorized<float> buffer_vec1 = data_vec1.abs().neg().exp();
        Vectorized<float> output_vec1 = min_vec - buffer_vec1.log1p();
        convert_from_float<scalar_t>(buffer_vec0, buffer_vec1).store(buffer_data + begin + d);
        convert_from_float<scalar_t>(output_vec0, output_vec1).store(output_data + begin + d);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
        Vectorized<float> data_vec0, data_vec1;
        std::tie(data_vec0, data_vec1) = convert_to_float<scalar_t>(data_vec);
        Vectorized<float> min_vec = minimum(data_vec0, Vectorized<float>(float(0)));
        Vectorized<float> buffer_vec0 = data_vec0.abs().neg().exp();
        Vectorized<float> output_vec0 = min_vec - buffer_vec0.log1p();
        min_vec = minimum(data_vec1, Vectorized<float>(float(0)));
        Vectorized<float> buffer_vec1 = data_vec1.abs().neg().exp();
        Vectorized<float> output_vec1 = min_vec - buffer_vec1.log1p();
        convert_from_float<scalar_t>(buffer_vec0, buffer_vec1).store(buffer_data + begin + d, size - d);
        convert_from_float<scalar_t>(output_vec0, output_vec1).store(output_data + begin + d, size - d);
      }
    });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&] {
      using Vec = Vectorized<scalar_t>;
      scalar_t* output_data = output.data_ptr<scalar_t>();
      scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
      scalar_t* input_data = input.data_ptr<scalar_t>();
      parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
        int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec data_vec = Vec::loadu(input_data + begin+ d);
          Vec min_vec = vec::minimum(data_vec, Vec(scalar_t(0)));
          Vec buffer_vec = data_vec.abs().neg().exp();
          Vec output_vec = min_vec - buffer_vec.log1p();
          buffer_vec.store(buffer_data + begin + d);
          output_vec.store(output_data + begin + d);
        }
        if (size - d > 0) {
          Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
          Vec min_vec = vec::minimum(data_vec, Vec(scalar_t(0)));
          Vec buffer_vec = data_vec.abs().neg().exp();
          Vec output_vec = min_vec - buffer_vec.log1p();
          buffer_vec.store(buffer_data + begin + d, size - d);
          output_vec.store(output_data + begin + d, size - d);
        }
      });
    });
  }
}

static void log_sigmoid_backward_cpu_kernel(TensorIterator& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      auto zero_val = float(0);
      auto zero_vec = Vectorized<float>(zero_val);
      auto one_val = float(1);
      auto one_vec = Vectorized<float>(one_val);
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          auto in_negative = float(a) < float(0);
          auto max_deriv = in_negative ? float(1) : float(0);
          auto sign = in_negative ? float(1) : -float(1);
          return (max_deriv - sign * (float(b) / (float(1) + b))) * float(c);
        },
        [=](Vec a, Vec b, Vec c) -> Vec {
          Vectorized<float> a0, a1, b0, b1, c0, c1;
          std::tie(a0, a1) = convert_to_float<scalar_t>(a);
          std::tie(b0, b1) = convert_to_float<scalar_t>(b);
          std::tie(c0, c1) = convert_to_float<scalar_t>(c);
          auto mask = a0 < zero_vec;
          auto max_deriv_vec = Vectorized<float>::blendv(zero_vec, one_vec, mask);
          auto sign_vec = Vectorized<float>::blendv(one_vec.neg(), one_vec, mask);
          a0 = (max_deriv_vec - sign_vec * (b0 / (one_vec + b0))) * c0;
          mask = a1 < zero_vec;
          max_deriv_vec = Vectorized<float>::blendv(zero_vec, one_vec, mask);
          sign_vec = Vectorized<float>::blendv(one_vec.neg(), one_vec, mask);
          a1 = (max_deriv_vec - sign_vec * (b1 / (one_vec + b1))) * c1;
          return convert_from_float<scalar_t>(a0, a1);
        });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto zero_val = scalar_t(0);
    auto zero_vec = Vec(zero_val);
    auto one_val = scalar_t(1);
    auto one_vec = Vec(one_val);
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        auto in_negative = a < scalar_t(0);
        auto max_deriv = in_negative ? scalar_t(1) : scalar_t(0);
        auto sign = in_negative ? scalar_t(1) : -scalar_t(1);
        return (max_deriv - sign * (b / (scalar_t(1) + b))) * c;
      },
      [=](Vec a, Vec b, Vec c) -> Vec {
        auto mask = a < zero_vec;
        auto max_deriv_vec = Vec::blendv(zero_vec, one_vec, mask);
        auto sign_vec = Vec::blendv(one_vec.neg(), one_vec, mask);
        return (max_deriv_vec - sign_vec * (b / (one_vec + b))) * c;
      });
  });
  }
}

static void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold_scalar,
    const Scalar& value_scalar) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "threshold_cpu", [&]() {
      using Vec = Vectorized<float>;
      float threshold = threshold_scalar.to<float>();
      Vec threshold_v = Vec(threshold);
      scalar_t value = value_scalar.to<scalar_t>();
      Vec value_v = Vec(float(value));
      cpu_kernel_vec(
          iter,
          [&](scalar_t x, scalar_t other) -> scalar_t {
            return float(x) <= threshold ? value : other;
          },
          [&](Vectorized<scalar_t> x, Vectorized<scalar_t> other) -> Vectorized<scalar_t> {
            Vec x0, x1, other0, other1;
            std::tie(x0, x1) = convert_to_float<scalar_t>(x);
            std::tie(other0, other1) = convert_to_float<scalar_t>(other);
            return convert_from_float<scalar_t>(Vec::blendv(other0, value_v, x0 <= threshold_v),
                                                Vec::blendv(other1, value_v, x1 <= threshold_v));
          });
    });
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "threshold_cpu", [&] {
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
}

void elu_kernel(TensorIteratorBase& it, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  if (at::isReducedFloatingType(it.common_dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "elu_cpu", [&]() {
      auto negcoef = alpha.to<float>() * scale.to<float>();
      auto poscoef = scale.to<float>();
      auto negiptcoef = input_scale.to<float>();
      const Vectorized<float> negcoef_vec(negcoef);
      const Vectorized<float> negiptcoef_vec(negiptcoef);
      const Vectorized<float> poscoef_vec(poscoef);
      const Vectorized<float> one_vec(static_cast<float>(1));
      const Vectorized<float> zero_vec(static_cast<float>(0));
      cpu_kernel_vec(
        it,
        [negcoef, negiptcoef, poscoef](scalar_t a) -> scalar_t {
          return float(a) <= float(0) ? (std::exp(float(a) * negiptcoef) - float(1)) * negcoef : float(a) * poscoef;
        },
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &one_vec, &zero_vec](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_to_float<scalar_t>(a);
          auto cmp0 = (a0 > zero_vec);
          auto cmp1 = (a1 > zero_vec);
          if (!cmp0.zero_mask() && !cmp1.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
            return convert_from_float<scalar_t>(a0 * poscoef_vec, a1 * poscoef_vec);
          } else {
            auto res0 = Vectorized<float>::blendv(((a0 * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a0 * poscoef_vec, cmp0);
            auto res1 = Vectorized<float>::blendv(((a1 * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a1 * poscoef_vec, cmp1);
            return convert_from_float<scalar_t>(res0, res1);
          }
        });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(it.common_dtype(), "elu_cpu", [&]() {
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
}

void elu_backward_kernel(TensorIteratorBase& it, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, bool is_result) {
  if (at::isReducedFloatingType(it.common_dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "elu_backward_cpu", [&]() {
    auto negcoef = alpha.to<float>() * scale.to<float>();
    auto poscoef = scale.to<float>();
    auto negiptcoef = input_scale.to<float>();
    const Vectorized<float> negcoef_vec(negcoef);
    const Vectorized<float> negiptcoef_vec(negiptcoef);
    const Vectorized<float> poscoef_vec(poscoef);
    const Vectorized<float> zero_vec(static_cast<float>(0));
    cpu_kernel_vec(
        it,
        [negcoef, negiptcoef, poscoef, is_result](scalar_t a, scalar_t b) -> scalar_t {
          if (is_result) {
            return float(b) <= float(0) ? float(a) * negiptcoef * (float(b) + negcoef) : float(a) * poscoef;
          } else {
            return float(b) <= float(0) ? float(a) * negiptcoef * negcoef * std::exp(float(b) * negiptcoef): float(a) * poscoef;
          }
        },
        [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &zero_vec, is_result](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_to_float<scalar_t>(a);
          Vectorized<float> b0, b1;
          std::tie(b0, b1) = convert_to_float<scalar_t>(b);
          auto cmp0 = (b0 > zero_vec);
          auto cmp1 = (b1 > zero_vec);
          if (is_result) {
            if (!cmp0.zero_mask() && !cmp1.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
              return convert_from_float<scalar_t>(a0 * poscoef_vec, a1 * poscoef_vec);
            } else {
              auto res0 = Vectorized<float>::blendv(a0 * negiptcoef_vec * (b0 + negcoef_vec), a0 * poscoef_vec, cmp0);
              auto res1 = Vectorized<float>::blendv(a1 * negiptcoef_vec * (b1 + negcoef_vec), a1 * poscoef_vec, cmp1);
              return convert_from_float<scalar_t>(res0, res1);
            }
          } else {
            auto res0 = Vectorized<float>::blendv(a0 * negiptcoef_vec * negcoef_vec * (b0 * negiptcoef_vec).exp(), a0 * poscoef_vec, cmp0);
            auto res1 = Vectorized<float>::blendv(a1 * negiptcoef_vec * negcoef_vec * (b1 * negiptcoef_vec).exp(), a1 * poscoef_vec, cmp1);
            return convert_from_float<scalar_t>(res0, res1);
          }
        });
    });
  } else {
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
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.
void GeluKernelImpl(TensorIteratorBase& it, GeluType approximate) {
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
  if (approximate == GeluType::Tanh) {
    if (at::isReducedFloatingType(it.common_dtype())) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "GeluKernelImpl", [&]() {
        auto kBetaVec = Vectorized<float>((float)(M_SQRT2 * M_2_SQRTPI * 0.5));
        auto kKappaVec = Vectorized<float>((float)(0.044715));
        auto kOneVec = Vectorized<float>((float)(1));
        auto kPointFiveVec = Vectorized<float>((float)(0.5));
        cpu_kernel_vec(
            it,
            [](scalar_t x) -> scalar_t {
              const float kBeta = float(M_SQRT2 * M_2_SQRTPI * 0.5);
              const float kKappa = float(0.044715);
              float x_cube = float(x) * float(x) * float(x);
              float inner = kBeta * (float(x) + kKappa * x_cube);
              return float(0.5) * float(x) * (float(1) + std::tanh(inner));
            },
            [&](Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
              Vectorized<float> x0, x1;
              std::tie(x0, x1) = convert_to_float<scalar_t>(x);
              auto x0_cube = x0 * x0 * x0;
              auto x1_cube = x1 * x1 * x1;
              auto inner_vec0 = kBetaVec * (x0 + kKappaVec * x0_cube);
              auto inner_vec1 = kBetaVec * (x1 + kKappaVec * x1_cube);
              auto res0 = kPointFiveVec * x0 * (kOneVec + inner_vec0.tanh());
              auto res1 = kPointFiveVec * x1 * (kOneVec + inner_vec1.tanh());
              return convert_from_float<scalar_t>(res0, res1);
            },
            grain_size);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluKernelImpl", [&]() {
        using Vec = vec::Vectorized<scalar_t>;
        const Vec kBetaVec(scalar_t(M_SQRT2 * M_2_SQRTPI * 0.5));
        const Vec kKappaVec(scalar_t(0.044715));
        const Vec kOneVec(scalar_t(1));
        const Vec kPointFiveVec(scalar_t(0.5));
        cpu_kernel_vec(
            it,
            [](scalar_t x) {
              const scalar_t kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
              const scalar_t kKappa = 0.044715;
              auto x_cube = x * x * x;
              auto inner = kBeta * (x + kKappa * x_cube);
              return scalar_t(0.5) * x * (scalar_t(1) + std::tanh(inner));
            },
            [&](Vec x_vec) {
              auto x_cube = x_vec * x_vec * x_vec;
              auto inner_vec = kBetaVec * (x_vec + kKappaVec * x_cube);
              return kPointFiveVec * x_vec * (kOneVec + inner_vec.tanh());
            },
            grain_size);
      });
    }
  } else {
    if (at::isReducedFloatingType(it.common_dtype())) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
        auto kAlphaVec = Vectorized<float>((float)(M_SQRT1_2));
        auto kOneVec = Vectorized<float>((float)(1));
        auto kPointFiveVec = Vectorized<float>((float)(0.5));
        cpu_kernel_vec(
            it,
            [](scalar_t x) -> scalar_t {
              const float kAlpha = float(M_SQRT1_2);
              return float(x) * float(0.5) * (float(1) + std::erf(float(x) * kAlpha));
            },
            [&](Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
              Vectorized<float> x0, x1;
              std::tie(x0, x1) = convert_to_float<scalar_t>(x);
              auto res0 = x0 * kPointFiveVec * (kOneVec + (x0 * kAlphaVec).erf());
              auto res1 = x1 * kPointFiveVec * (kOneVec + (x1 * kAlphaVec).erf());
              return convert_from_float<scalar_t>(res0, res1);
            },
            grain_size);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluKernelImpl", [&]() {
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
  }
}

void GeluBackwardKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    if (at::isReducedFloatingType(it.common_dtype())) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "GeluBackwardKernelImpl", [&]() {
      auto kBetaVec = Vectorized<float>((float)(M_SQRT2 * M_2_SQRTPI * 0.5));
      auto kKappaVec = Vectorized<float>((float)(0.044715));
      auto kOneVec = Vectorized<float>((float)(1));
      auto kThreeVec = Vectorized<float>((float)(3));
      auto kPointFiveVec = Vectorized<float>((float)(0.5));
      cpu_kernel_vec(
          it,
          [](scalar_t dy, scalar_t x) -> scalar_t {
            const float kBeta = float(M_SQRT2 * M_2_SQRTPI * 0.5);
            const float kKappa = float(0.044715);
            float x_sq = float(x) * float(x);
            float x_cube = x_sq * float(x);
            float inner = kBeta * (float(x) + kKappa * x_cube);
            float tanh_inner = float(std::tanh(inner));

            float left = float(0.5) * float(x);
            float right = float(1) + tanh_inner;

            float left_derivative = float(0.5) * right;

            float tanh_derivative = float(1) - tanh_inner * tanh_inner;
            float inner_derivative =
              kBeta * (float(1) + float(3) * kKappa * x_sq);
            float right_derivative = left * tanh_derivative * inner_derivative;

            return float(dy) * (left_derivative + right_derivative);
          },
          [&](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            Vectorized<float> x0_vec, x1_vec;
            std::tie(x0_vec, x1_vec) = convert_to_float<scalar_t>(x_vec);
            Vectorized<float> dy0_vec, dy1_vec;
            std::tie(dy0_vec, dy1_vec) = convert_to_float<scalar_t>(dy_vec);
            auto x0_sq = x0_vec * x0_vec;
            auto x1_sq = x1_vec * x1_vec;
            auto x0_cube = x0_vec * x0_vec * x0_vec;
            auto x1_cube = x1_vec * x1_vec * x1_vec;
            auto inner_vec0 = kBetaVec * (x0_vec + kKappaVec * x0_cube);
            auto inner_vec1 = kBetaVec * (x1_vec + kKappaVec * x1_cube);
            auto tanh_inner_vec0 = inner_vec0.tanh();
            auto tanh_inner_vec1 = inner_vec1.tanh();

            auto left_vec0 = kPointFiveVec * x0_vec;
            auto left_vec1 = kPointFiveVec * x1_vec;
            auto right_vec0 = kOneVec + tanh_inner_vec0;
            auto right_vec1 = kOneVec + tanh_inner_vec1;

            auto left_derivative_vec0 = kPointFiveVec * right_vec0;
            auto left_derivative_vec1 = kPointFiveVec * right_vec1;

            auto tanh_derivative_vec0 = kOneVec - tanh_inner_vec0 * tanh_inner_vec0;
            auto tanh_derivative_vec1 = kOneVec - tanh_inner_vec1 * tanh_inner_vec1;
            auto inner_derivative_vec0 = kBetaVec * (kOneVec + kThreeVec * kKappaVec * x0_sq);
            auto inner_derivative_vec1 = kBetaVec * (kOneVec + kThreeVec * kKappaVec * x1_sq);
            auto right_derivative_vec0 = left_vec0 * tanh_derivative_vec0 * inner_derivative_vec0;
            auto right_derivative_vec1 = left_vec1 * tanh_derivative_vec1 * inner_derivative_vec1;

            auto res0 = dy0_vec * (left_derivative_vec0 + right_derivative_vec0);
            auto res1 = dy1_vec * (left_derivative_vec1 + right_derivative_vec1);
            return convert_from_float<scalar_t>(res0, res1);
          });
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluBackwardKernelImpl", [&]() {
        using Vec = vec::Vectorized<scalar_t>;
        const Vec kBetaVec(scalar_t(M_SQRT2 * M_2_SQRTPI * 0.5));
        const Vec kKappaVec(scalar_t(0.044715));
        const Vec kOneVec(scalar_t(1));
        const Vec kThreeVec(scalar_t(3));
        const Vec kPointFiveVec(scalar_t(0.5));
        cpu_kernel_vec(
            it,
            [](scalar_t dy, scalar_t x) {
              const scalar_t kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
              const scalar_t kKappa = 0.044715;
              auto x_sq = x * x;
              auto x_cube = x_sq * x;
              auto inner = kBeta * (x + kKappa * x_cube);
              auto tanh_inner = std::tanh(inner);

              auto left = scalar_t(0.5) * x;
              auto right = scalar_t(1) + tanh_inner;

              auto left_derivative = scalar_t(0.5) * right;

              auto tanh_derivative = scalar_t(1) - tanh_inner * tanh_inner;
              auto inner_derivative =
                kBeta * (scalar_t(1) + scalar_t(3) * kKappa * x_sq);
              auto right_derivative = left * tanh_derivative * inner_derivative;

              return dy * (left_derivative + right_derivative);
            },
            [&](Vec dy_vec, Vec x_vec) {
              auto x_sq = x_vec * x_vec;
              auto x_cube = x_vec * x_vec * x_vec;
              auto inner_vec =
                  kBetaVec * (x_vec + kKappaVec * x_cube);
              auto tanh_inner_vec = inner_vec.tanh();

              auto left_vec = kPointFiveVec * x_vec;
              auto right_vec = kOneVec + tanh_inner_vec;

              auto left_derivative_vec = kPointFiveVec * right_vec;

              auto tanh_derivative_vec =
                  kOneVec - tanh_inner_vec * tanh_inner_vec;
              auto inner_derivative_vec =
                  kBetaVec * (kOneVec + kThreeVec * kKappaVec * x_sq);
              auto right_derivative_vec =
                  left_vec * tanh_derivative_vec * inner_derivative_vec;

              return dy_vec * (left_derivative_vec + right_derivative_vec);
            });
      });
    }
  } else {
    if (at::isReducedFloatingType(it.common_dtype())) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(it.common_dtype(), "GeluBackwardKernelImpl", [&]() {
      auto kAlphaVec = Vectorized<float>((float)(M_SQRT1_2));
      auto kBetaVec = Vectorized<float>((float)(M_2_SQRTPI * M_SQRT1_2 * 0.5));
      auto kOneVec = Vectorized<float>((float)(1));
      auto kPointFiveVec = Vectorized<float>((float)(0.5));
      auto kMinusPointFiveVec = Vectorized<float>((float)(-0.5));
      cpu_kernel_vec(
          it,
          [](scalar_t dy, scalar_t x) -> scalar_t {
              const float kAlpha = float(M_SQRT1_2);
              const float kBeta = float(M_2_SQRTPI) * float(M_SQRT1_2) * float(0.5);
              const float cdf =
                  float(0.5) * (float(1) + std::erf(float(x) * kAlpha));
              const float pdf = kBeta * std::exp(float(x) * float(x) * float(-0.5));
              return float(dy) * (cdf + float(x) * pdf);
          },
          [&](Vectorized<scalar_t> dy, Vectorized<scalar_t> x) -> Vectorized<scalar_t> {
              Vectorized<float> x0, x1;
              std::tie(x0, x1) = convert_to_float<scalar_t>(x);
              Vectorized<float> dy0, dy1;
              std::tie(dy0, dy1) = convert_to_float<scalar_t>(dy);
              auto cdf_vec0 = kPointFiveVec * (kOneVec + (x0 * kAlphaVec).erf());
              auto cdf_vec1 = kPointFiveVec * (kOneVec + (x1 * kAlphaVec).erf());
              auto pdf_vec0 = kBetaVec * (x0 * x0 * kMinusPointFiveVec).exp();
              auto pdf_vec1 = kBetaVec * (x1 * x1 * kMinusPointFiveVec).exp();
              auto res0 = dy0 * (cdf_vec0 + x0 * pdf_vec0);
              auto res1 = dy1 * (cdf_vec1 + x1 * pdf_vec1);
              return convert_from_float<scalar_t>(res0, res1);
          });
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(
          it.dtype(), "GeluBackwardKernelImpl", [&]() {
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
  }
}

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardsigmoid_cpu", [&]() {
    const float zero(0.0f);
    const float three(3.0f);
    const float six(6.0f);
    using Vec = vec::Vectorized<float>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    cpu_kernel_vec(
        iter,
        [&](scalar_t self_val) -> scalar_t {
          return std::min(std::max(float(self_val) + three, zero), six) / six;
        },
        [&](vec::Vectorized<scalar_t> self_val) -> vec::Vectorized<scalar_t> {
          Vectorized<float> self_val0, self_val1;
          std::tie(self_val0, self_val1) = convert_to_float<scalar_t>(self_val);
          self_val0 = minimum(
            maximum(self_val0 + kThreeVec, kZeroVec),
            kSixVec
          ) / kSixVec;
          self_val1 = minimum(
            maximum(self_val1 + kThreeVec, kZeroVec),
            kSixVec
          ) / kSixVec;
          return convert_from_float<scalar_t>(self_val0, self_val1);
        });
    });
  } else {
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
}

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.common_dtype(), "hardsigmoid_backward", [&]() {
    const float zero(0.0f);
    const float three(3.0f);
    const float neg_three(-3.0f);
    const float one_sixth(1.0f / 6.0f);
    using Vec = Vectorized<float>;
    Vec kZeroVec(0.0f);
    Vec kOneSixthVec(1.0f / 6.0f);
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) -> scalar_t {
          return (float(self_val) > neg_three && float(self_val) < three)
            ? float(grad_val) * one_sixth
            : zero;
        },
        [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          Vec self_val0, self_val1, grad_val0, grad_val1;
          std::tie(self_val0, self_val1) = convert_to_float<scalar_t>(self_val);
          std::tie(grad_val0, grad_val1) = convert_to_float<scalar_t>(grad_val);
          Vec gradNonZeroMask = (self_val0 > neg_three) & (self_val0 < three);
          self_val0 = Vec::blendv(kZeroVec, grad_val0 * kOneSixthVec, gradNonZeroMask);
          gradNonZeroMask = (self_val1 > neg_three) & (self_val1 < three);
          self_val1 = Vec::blendv(kZeroVec, grad_val1 * kOneSixthVec, gradNonZeroMask);
          return convert_from_float<scalar_t>(self_val0, self_val1);
        });
    });
  } else {
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
}

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : self_val;
        },
        [=](Vec self_val) {
          return Vec::blendv(self_val, Vec(0), (self_val >= -lambd_val) & (self_val <= lambd_val));
        });
  });
}

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.common_dtype(), "softshrink_cpu", [&]() {
    auto lambd_val = lambd.to<float>();
    auto lambdVec = Vectorized<float>(lambd_val);
    cpu_kernel_vec(
      iter,
      [=](scalar_t a) -> scalar_t {
        return float(a) > lambd_val ? a - lambd_val : (float(a) < -lambd_val ? a + lambd_val : float(0));
      },
      [=](Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          Vectorized<float> self_val0, self_val1;
          Vectorized<scalar_t> self_val_t0, self_val_t1;
          std::tie(self_val0, self_val1) = convert_to_float<scalar_t>(self_val);
          self_val_t0 = convert_from_float<scalar_t>((self_val0 > lambdVec) & (self_val0 - lambdVec), (self_val1 > lambdVec) & (self_val1 - lambdVec));
          self_val_t1 = convert_from_float<scalar_t>((self_val0 < -lambd_val) & (self_val0 + lambdVec), (self_val1 < -lambd_val) & (self_val1 + lambdVec));
          return (self_val_t0 | self_val_t1);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softshrink_cpu", [&]() {
    auto lambd_val = lambd.to<scalar_t>();
    auto lambdVec = Vectorized<scalar_t>(lambd_val);
    cpu_kernel_vec(
      iter,
      [=](scalar_t a) -> scalar_t {
        return a > lambd_val ? a - lambd_val : (a < -lambd_val ? a + lambd_val : scalar_t(0));
      },
      [=](Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
          Vectorized<scalar_t> self_val_t0, self_val_t1;
          self_val_t0 = (self_val > lambdVec) & (self_val - lambdVec);
          self_val_t1 = (self_val < -lambd_val) & (self_val + lambdVec);
          return (self_val_t0 | self_val_t1);
      });
  });
  }
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "shrink_backward_cpu", [&] {
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
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&]() {
      auto min_val = min.to<float>();
      auto max_val = max.to<float>();
      cpu_kernel_vec(
          iter,
          [=](scalar_t grad_val, scalar_t self_val) -> scalar_t {
            return (float(self_val) <= min_val || float(self_val) >= max_val) ? scalar_t(0) : grad_val;
          },
          [=](Vectorized<scalar_t> grad_val, Vectorized<scalar_t> self_val) -> Vectorized<scalar_t> {
            Vectorized<float> grad_val0, grad_val1, self_val0, self_val1;
            std::tie(grad_val0, grad_val1) = convert_to_float<scalar_t>(grad_val);
            std::tie(self_val0, self_val1) = convert_to_float<scalar_t>(self_val);
            return convert_from_float<scalar_t>(
              ((self_val0 > min_val) & (self_val0 < max_val)) & grad_val0,
              ((self_val1 > min_val) & (self_val1 < max_val)) & grad_val1
            );
          });
    });
  } else {
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
}

void hardswish_kernel(TensorIterator& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardswish_cpu", [&]() {
    const float zero(0.0f);
    const float three(3.0f);
    const float six(6.0f);
    using Vec = vec::Vectorized<float>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kSixVec(six);
    cpu_kernel_vec(
      iter,
      [&](scalar_t x) -> scalar_t {
        return float(x) * std::min(std::max(float(x) + three, zero), six) / six;
      },
      [&](vec::Vectorized<scalar_t> x_vec) {
        Vectorized<float> x_vec0, x_vec1;
        std::tie(x_vec0, x_vec1) = convert_to_float<scalar_t>(x_vec);
        x_vec0 = x_vec0 * minimum(
          maximum(x_vec0 + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
        x_vec1 = x_vec1 * minimum(
          maximum(x_vec1 + kThreeVec, kZeroVec),
          kSixVec
        ) / kSixVec;
        return convert_from_float<scalar_t>(x_vec0, x_vec1);
      });
    });
  } else {
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
}

void hardswish_backward_kernel(TensorIterator& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "hardswish_backward_cpu", [&]() {
    const float zero(0.0f);
    const float three(3.0f);
    const float neg_three(-3.0f);
    const float one_half(0.5f);
    using Vec = vec::Vectorized<float>;
    const Vec kZeroVec(zero);
    const Vec kThreeVec(three);
    const Vec kNegThreeVec(neg_three);
    const Vec kOneHalfVec(one_half);
    cpu_kernel_vec(
      iter,
      [&](scalar_t grad_val, scalar_t self_val) -> scalar_t {
        if (float(self_val) < neg_three) {
          return zero;
        } else if (float(self_val) <= three) {
          return float(grad_val) * ((float(self_val) / three) + one_half);
        } else {
          return grad_val;
        }
      },
      [&](vec::Vectorized<scalar_t> grad_val, vec::Vectorized<scalar_t> self_val) {
        Vectorized<float> self_val0, self_val1, grad_val0, grad_val1;
        std::tie(self_val0, self_val1) = convert_to_float<scalar_t>(self_val);
        std::tie(grad_val0, grad_val1) = convert_to_float<scalar_t>(grad_val);
        self_val0 = Vec::blendv(
          Vec::blendv(
            grad_val0 * ((self_val0 / kThreeVec) + kOneHalfVec),
            grad_val0,
            self_val0 >= kThreeVec
          ),
          kZeroVec,
          self_val0 < kNegThreeVec
        );
        self_val1 = Vec::blendv(
          Vec::blendv(
            grad_val1 * ((self_val1 / kThreeVec) + kOneHalfVec),
            grad_val1,
            self_val1 >= kThreeVec
          ),
          kZeroVec,
          self_val1 < kNegThreeVec
        );
        return convert_from_float<scalar_t>(self_val0, self_val1);
      });
    });
  } else {
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
}

static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&]() {
    auto zero_vec = Vectorized<float>((float)(0));
    auto one_vec = Vectorized<float>((float)(1));
    float negval = negval_.to<float>();
    Vectorized<float> negval_v = Vectorized<float>(negval);
    cpu_kernel_vec(
        iter,
        [&](scalar_t a) -> scalar_t {
          return float(a) > float(0) ? float(a) : float(a) * negval;
        },
        [&](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_to_float<scalar_t>(a);
          auto res0 = a0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
          auto res1 = a1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
          return convert_from_float<scalar_t>(res0, res1);
        });
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
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&]() {
    auto zero_vec = Vectorized<float>((float)(0));
    auto one_vec = Vectorized<float>((float)(1));
    float negval = negval_.to<float>();
    Vectorized<float> negval_v = Vectorized<float>(negval);
    cpu_kernel_vec(
      iter,
      [&](scalar_t a, scalar_t b) -> scalar_t {
        return float(a) > float(0) ? float(b) : float(b) * negval;
      },
      [&](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
        Vectorized<float> a0, a1, b0, b1;
        std::tie(a0, a1) = convert_to_float<scalar_t>(a);
        std::tie(b0, b1) = convert_to_float<scalar_t>(b);
        auto res0 = b0 * (Vectorized<float>::blendv(negval_v, one_vec, a0 > zero_vec));
        auto res1 = b1 * (Vectorized<float>::blendv(negval_v, one_vec, a1 > zero_vec));
        return convert_from_float<scalar_t>(res0, res1);
      });
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
    if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "softplus_cpu", [&]() {
      using Vec = Vectorized<float>;
      auto beta = beta_.to<float>();
      auto threshold = threshold_.to<float>();
      const Vec beta_vec(beta);
      const Vec threshold_vec(threshold);
      cpu_kernel_vec(
          iter,
          [beta, threshold](scalar_t a) -> scalar_t {
            return (float(a) * beta) > threshold ? a
              : static_cast<scalar_t>((std::log1p(std::exp(float(a) * beta))) / beta);
          },
          [beta_vec, threshold_vec](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
            Vectorized<float> a0, a1;
            std::tie(a0, a1) = convert_to_float<scalar_t>(a);
            a0 = Vec::blendv((a0 * beta_vec).exp().log1p() / beta_vec, a0, (a0 * beta_vec) > threshold_vec);
            a1 = Vec::blendv((a1 * beta_vec).exp().log1p() / beta_vec, a1, (a1 * beta_vec) > threshold_vec);
            return convert_from_float<scalar_t>(a0, a1);
          }
      );
    });
  } else {
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
}

void softplus_backward_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "softplus_backward_cpu", [&]() {
    using Vec = Vectorized<float>;
    auto beta = beta_.to<float>();
    auto threshold = threshold_.to<float>();
    const Vec beta_vec(beta);
    const Vec threshold_vec(threshold);
    const Vec one_vec(static_cast<float>(1.0));
    cpu_kernel_vec(
        iter,
        [beta, threshold](scalar_t a, scalar_t b) -> scalar_t {
          float z = std::exp(float(b) * beta);
          return (float(b) * beta) > threshold ? a : static_cast<scalar_t>(float(a) * z / (z + float(1.)));
        },
        [beta_vec, one_vec, threshold_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          Vectorized<float> a0, a1, b0, b1;
          std::tie(a0, a1) = convert_to_float<scalar_t>(a);
          std::tie(b0, b1) = convert_to_float<scalar_t>(b);
          Vec z = (b0 * beta_vec).exp();
          a0 = Vec::blendv(a0 * z / (z + one_vec), a0, (b0 * beta_vec) > threshold_vec);
          z = (b1 * beta_vec).exp();
          a1 = Vec::blendv(a1 * z / (z + one_vec), a1, (b1 * beta_vec) > threshold_vec);
          return convert_from_float<scalar_t>(a0, a1);
        });
    });
  } else {
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
}

void glu_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&]() {
    const float float_one_val(1);
    const Vectorized<float> float_one_vec(float_one_val);
    cpu_kernel_vec(
      iter,
      [float_one_val](scalar_t a, scalar_t b) -> scalar_t {
        return float(a) * (float_one_val / (float_one_val + std::exp(- float(b))));
      },
      [float_one_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
        Vectorized<float> a0, a1, b0, b1;
        std::tie(a0, a1) = convert_to_float<scalar_t>(a);
        std::tie(b0, b1) = convert_to_float<scalar_t>(b);
        return convert_from_float<scalar_t>(a0 * (float_one_vec / (float_one_vec + b0.neg().exp())),
                                            a1 * (float_one_vec / (float_one_vec + b1.neg().exp())));
      });
    });
  } else {
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
}

void glu_jvp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_jvp_cpu", [&] {
    using Vec = Vectorized<scalar_t>;
    const scalar_t one(1);
    const Vec ones(one);
    cpu_kernel_vec(
      iter,
      [one](scalar_t res, scalar_t b, scalar_t da, scalar_t db) -> scalar_t {
        const auto sig_b = one / (one + std::exp(-b));
        return da * sig_b + res * (db - sig_b * db);
      },
      [ones](Vec res, Vec b, Vec da, Vec db) -> Vec {
        const auto sig_b = ones / (ones + b.neg().exp());
        return da * sig_b + res * (db - sig_b * db);
      }
    );
  });
}

void glu_backward_kernel(TensorIterator& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "glu_backward_cpu", [&]() {
    const float float_one_val(1);
    const Vectorized<float> float_one_vec(float_one_val);
    cpu_kernel_vec(
      iter,
      [float_one_val](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return  (float_one_val - float(a)) * float(a) * float(b) * float(c);
      },
      [float_one_vec](Vectorized<scalar_t> a, Vectorized<scalar_t> b, Vectorized<scalar_t> c) -> Vectorized<scalar_t> {
        Vectorized<float> a0, a1, b0, b1, c0, c1;
        std::tie(a0, a1) = convert_to_float<scalar_t>(a);
        std::tie(b0, b1) = convert_to_float<scalar_t>(b);
        std::tie(c0, c1) = convert_to_float<scalar_t>(c);
        a0 = (float_one_vec - a0) * a0 * b0 * c0;
        a1 = (float_one_vec - a1) * a1 * b1 * c1;
        return convert_from_float<scalar_t>(a0, a1);
      });
    });
  } else {
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
}

void silu_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "silu_cpu", [&]() {
      const Vectorized<float> kOneVec(1.0f);
      cpu_kernel_vec(
          iter,
          [](scalar_t x) -> scalar_t {
            return float(x) / (1.0f + std::exp(-float(x)));
          },
          [kOneVec](Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
            Vectorized<float> x_vec0, x_vec1;
            std::tie(x_vec0, x_vec1) = convert_to_float<scalar_t>(x_vec);
            return convert_from_float<scalar_t>(
              x_vec0 / (kOneVec + x_vec0.neg().exp()),
              x_vec1 / (kOneVec + x_vec1.neg().exp()));
          });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      iter.dtype(), "silu_cpu", [&]() {
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
}

void silu_backward_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "silu_backward_cpu", [&]() {
    const Vectorized<float> kOneVec(1.0f);
    cpu_kernel_vec(
        iter,
        [](scalar_t dy, scalar_t x) -> scalar_t {
          const float sigmoid =
              1.0f / (1.0f + std::exp(-float(x)));
          return dy * sigmoid * (1.0f + x * (1.0f - sigmoid));
        },
        [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
          Vectorized<float> x_vec0, x_vec1, dy_vec0, dy_vec1;
          std::tie(x_vec0, x_vec1) = convert_to_float<scalar_t>(x_vec);
          std::tie(dy_vec0, dy_vec1) = convert_to_float<scalar_t>(dy_vec);
          const Vectorized<float> sigmoid0 =
              kOneVec / (kOneVec + x_vec0.neg().exp());
          const Vectorized<float> sigmoid1 =
              kOneVec / (kOneVec + x_vec1.neg().exp());
          return convert_from_float<scalar_t>(
            dy_vec0 * sigmoid0 * (kOneVec + x_vec0 * (kOneVec - sigmoid0)),
            dy_vec1 * sigmoid1 * (kOneVec + x_vec1 * (kOneVec - sigmoid1)));
        });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      iter.dtype(), "silu_backward_cpu", [&]() {
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
}

void mish_kernel(TensorIteratorBase& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "mish_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t{
          return static_cast<scalar_t>(float(x) * std::tanh(std::log1p(std::exp(float(x)))));
        },
        [](Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
          Vectorized<float> x_vec0, x_vec1;
          std::tie(x_vec0, x_vec1) = convert_to_float<scalar_t>(x_vec);
          return convert_from_float<scalar_t>(
            x_vec0 * x_vec0.exp().log1p().tanh(),
            x_vec1 * x_vec1.exp().log1p().tanh()
          );
        });
    });
  } else {
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
}

void mish_backward_kernel(TensorIterator& iter) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "mish_backward_cpu", [&]() {
    using Vec = Vectorized<float>;
    const Vec kOneVec(1.0f);
    cpu_kernel_vec(
        iter,
        [](scalar_t dy, scalar_t x) -> scalar_t {
          const float sigmoid =
              1.0f / (1.0f + std::exp(-float(x)));
          const float tanh_softplus = std::tanh(std::log1p(std::exp(float(x))));
          return dy * (tanh_softplus + x * sigmoid * (1.0f - tanh_softplus * tanh_softplus));
        },
        [kOneVec](Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) -> Vectorized<scalar_t> {
          Vectorized<float> x_vec0, x_vec1, dy_vec0, dy_vec1;
          std::tie(x_vec0, x_vec1) = convert_to_float<scalar_t>(x_vec);
          std::tie(dy_vec0, dy_vec1) = convert_to_float<scalar_t>(dy_vec);
          const Vec sigmoid0 = kOneVec / (kOneVec + x_vec0.neg().exp());
          const Vec sigmoid1 = kOneVec / (kOneVec + x_vec1.neg().exp());
          const Vec tanh_softplus0 = x_vec0.exp().log1p().tanh();
          const Vec tanh_softplus1 = x_vec1.exp().log1p().tanh();
          return convert_from_float<scalar_t>(
            dy_vec0 * (tanh_softplus0 + x_vec0 * sigmoid0 * (kOneVec - tanh_softplus0 * tanh_softplus0)),
            dy_vec1 * (tanh_softplus1 + x_vec1 * sigmoid1 * (kOneVec - tanh_softplus1 * tanh_softplus1))
          );
        });
    });
  } else {
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
}

void prelu_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
      iter,
      [](scalar_t input, scalar_t weight) {
        return (input > scalar_t(0)) ? input : weight * input;
      },
      [](Vec input, Vec weight) {
        return Vec::blendv(weight * input, input, input > Vec(0));
      });
  });
}

void prelu_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_cpu", [&]() {
    cpu_kernel_multiple_outputs(iter,
      [](scalar_t input, scalar_t weight, scalar_t grad) -> std::tuple<scalar_t, scalar_t> {
        auto mask = input > scalar_t{0};
        auto grad_input = mask ? grad : weight * grad;
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        return {grad_input, grad_weight};
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
REGISTER_DISPATCH(glu_jvp_stub, &glu_jvp_kernel);
REGISTER_DISPATCH(silu_stub, &silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, &silu_backward_kernel);
REGISTER_DISPATCH(mish_stub, &mish_kernel);
REGISTER_DISPATCH(mish_backward_stub, &mish_backward_kernel);
REGISTER_DISPATCH(prelu_stub, &prelu_kernel);
REGISTER_DISPATCH(prelu_backward_stub, &prelu_backward_kernel);

} // namespace at::native
