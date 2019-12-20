#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif // AT_MKL_ENABLED()

namespace at {
namespace native {

namespace {

static void threshold_kernel(
    TensorIterator& iter,
    Scalar threshold_scalar,
    Scalar value_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "threshold_cpu", [&] {
    using Vec = Vec256<scalar_t>;
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

#if AT_MKL_ENABLED()

template <typename T>
void MKLCdfNorm(int64_t N, const T* X, T* Y);

template <>
void MKLCdfNorm<float>(int64_t N, const float* X, float* Y) {
  vsCdfNorm(N, X, Y);
}

template <>
void MKLCdfNorm<double>(int64_t N, const double* X, double* Y) {
  vdCdfNorm(N, X, Y);
}

template <typename T>
void MKLMul(int64_t N, const T* A, const T* B, T* Y);

template <>
void MKLMul<float>(int64_t N, const float* A, const float* B, float* Y) {
  vsMul(N, A, B, Y);
}

template <>
void MKLMul<double>(int64_t N, const double* A, const double* B, double* Y) {
  vdMul(N, A, B, Y);
}

template <typename T>
void MKLExp(int64_t N, const T* X, T* Y);

template <>
void MKLExp<float>(int64_t N, const float* X, float* Y) {
  vsExp(N, X, Y);
}

template <>
void MKLExp<double>(int64_t N, const double* X, double* Y) {
  vdExp(N, X, Y);
}

template <typename T>
void GeluMKLKernelImpl(TensorIterator* it) {
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      GeluMKLKernelImpl<T>(&sub_it);
    }
    return;
  }
  const int64_t N = it->numel();
  const T* X_data = static_cast<T*>(it->data_ptr(1));
  T* Y_data = static_cast<T*>(it->data_ptr(0));
  MKLCdfNorm<T>(N, X_data, Y_data);
  MKLMul<T>(N, X_data, Y_data, Y_data);
}

template <typename T>
void GeluBackwardMKLKernelImpl(TensorIterator* it) {
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      GeluBackwardMKLKernelImpl<T>(&sub_it);
    }
    return;
  }
  constexpr T kBeta = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
  const int64_t N = it->numel();
  const T* dY_data = static_cast<T*>(it->data_ptr(1));
  const T* X_data = static_cast<T*>(it->data_ptr(2));
  T* dX_data = static_cast<T*>(it->data_ptr(0));
  Tensor cdf = at::empty({N}, it->input(1).options());
  T* cdf_data = cdf.template data_ptr<T>();
  MKLCdfNorm<T>(N, X_data, cdf_data);
  for (int64_t i = 0; i < N; ++i) {
    dX_data[i] = T(-0.5) * X_data[i] * X_data[i];
  }
  MKLExp(N, dX_data, dX_data);
  for (int64_t i = 0; i < N; ++i) {
    dX_data[i] = dY_data[i] * (cdf_data[i] + kBeta * X_data[i] * dX_data[i]);
  }
}

#else // AT_MKL_ENABLED()

template <typename T>
void GeluMKLKernelImpl(TensorIterator* /* it */) {
  AT_ASSERTM(false, "ATen not compiled with MKL");
}

template <typename T>
void GeluBackwardMKLKernelImpl(TensorIterator* /* it */) {
  AT_ASSERTM(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED()

void elu_kernel(TensorIterator& it, Scalar alpha, Scalar scale, Scalar input_scale) {
  AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_cpu", [&]() {
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    cpu_kernel(it, [=](scalar_t a) -> scalar_t {
      return a <= scalar_t(0) ? (std::exp(a * negiptcoef) - scalar_t(1)) * negcoef : a * poscoef;
    });
  });
}

void elu_backward_kernel(TensorIterator& it, Scalar alpha, Scalar scale, Scalar input_scale) {
  AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_backward_cpu", [&]() {
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    cpu_kernel(it, [=](scalar_t a, scalar_t b) -> scalar_t {
      return b <= scalar_t(0) ? a * negiptcoef * (b + negcoef) : a * poscoef;
    });
  });
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.
void GeluKernelImpl(TensorIterator& it) {
  if (at::hasMKL() && it.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
      GeluMKLKernelImpl<scalar_t>(&it);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
      using Vec = vec256::Vec256<scalar_t>;
      const Vec kAlphaVec(M_SQRT1_2);
      const Vec kOneVec(1);
      const Vec kPointFiveVec(0.5);
      cpu_kernel_vec(
          it,
          [](scalar_t x) {
            constexpr scalar_t kAlpha = M_SQRT1_2;
            return x * scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
          },
          [&](Vec x_vec) {
            return x_vec * kPointFiveVec *
                (kOneVec + (x_vec * kAlphaVec).erf());
          });
    });
  }
}

void GeluBackwardKernelImpl(TensorIterator& it) {
  if (hasMKL() && it.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluBackwardKernelImpl", [&]() {
      GeluBackwardMKLKernelImpl<scalar_t>(&it);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluBackwardKernelImpl", [&]() {
      using Vec = vec256::Vec256<scalar_t>;
      const Vec kAlphaVec(M_SQRT1_2);
      const Vec kBetaVec(M_2_SQRTPI * M_SQRT1_2 * 0.5);
      const Vec kOneVec(1);
      const Vec kPointFiveVec(0.5);
      const Vec kMinusPointFiveVec(-0.5);
      cpu_kernel_vec(
          it,
          [](scalar_t dy, scalar_t x) {
            constexpr scalar_t kAlpha = M_SQRT1_2;
            constexpr scalar_t kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
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

void hardshrink_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : self_val;
        },
        [=](Vec256<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & self_val;
        });
  });
}

void softshrink_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "softshrink_cpu", [&]() {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
      return a > lambd_val ? a - lambd_val : (a < -lambd_val ? a + lambd_val : scalar_t(0));
    });
  });
}

void shrink_backward_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "shrink_backward_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : grad_val;
        },
        [=](Vec256<scalar_t> grad_val, Vec256<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & grad_val;
        });
  });
}

void hardtanh_backward_kernel(TensorIterator& iter, Scalar min, Scalar max) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "hardshrink_backward_cpu", [&] {
    auto min_val = min.to<scalar_t>();
    auto max_val = max.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t grad_val, scalar_t self_val) {
          return (self_val <= min_val || self_val >= max_val) ? scalar_t(0) : grad_val;
        },
        [=](Vec256<scalar_t> grad_val, Vec256<scalar_t> self_val) {
          return ((self_val > min_val) & (self_val < max_val)) & grad_val;
        });
  });
}

static void leaky_relu_kernel(TensorIterator& iter, Scalar negval_) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&] {
    using Vec = Vec256<scalar_t>;
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

static void leaky_relu_backward_kernel(TensorIterator& iter, Scalar negval_) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&] {
    using Vec = Vec256<scalar_t>;
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

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
REGISTER_DISPATCH(elu_stub, &elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel);
REGISTER_DISPATCH(GeluKernel, &GeluKernelImpl);
REGISTER_DISPATCH(GeluBackwardKernel, &GeluBackwardKernelImpl);
REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel);
REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel);
REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel);

} // namespace native
} // namespace at
