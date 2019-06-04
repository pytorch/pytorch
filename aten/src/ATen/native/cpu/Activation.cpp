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

#if AT_MKL_ENABLED()

template <typename T>
void MKLCdfNorm(int N, const T* X, T* Y);

template <>
void MKLCdfNorm<float>(int N, const float* X, float* Y) {
  vsCdfNorm(N, X, Y);
}

template <>
void MKLCdfNorm<double>(int N, const double* X, double* Y) {
  vdCdfNorm(N, X, Y);
}

template <typename T>
void MKLMul(int N, const T* A, const T* B, T* Y);

template <>
void MKLMul<float>(int N, const float* A, const float* B, float* Y) {
  vsMul(N, A, B, Y);
}

template <>
void MKLMul<double>(int N, const double* A, const double* B, double* Y) {
  vdMul(N, A, B, Y);
}

template <typename T>
void GeluKernelMKLImpl(TensorIterator* it) {
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      GeluKernelMKLImpl<T>(&sub_it);
    }
    return;
  }
  const int N = it->numel();
  const T* X_data = static_cast<T*>(it->data_ptr(1));
  T* Y_data = static_cast<T*>(it->data_ptr(0));
  MKLCdfNorm<T>(N, X_data, Y_data);
  MKLMul<T>(N, X_data, Y_data, Y_data);
}

#else // AT_MKL_ENABLED()

template <typename T>
void GeluKernelMKLImpl(TensorIterator* /* it */) {
  AT_ASSERTM(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED()

template <typename T>
void GeluKernelImplInternal(TensorIterator* it) {
  using Vec = Vec256<T>;
  const Vec kOne = Vec(1.0);
  const Vec kPointFive = Vec(0.5);
  const Vec kSqrtPointFive = Vec(M_SQRT1_2);
  unary_kernel_vec(
      *it,
      [](T x) {
        return T(0.5) * x * (T(1) + std::erf(x * static_cast<T>(M_SQRT1_2)));
      },
      [&](const Vec& x) {
        return kPointFive * x * (kOne + (x * kSqrtPointFive).erf());
      });
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.
void GeluKernelImpl(TensorIterator* it) {
  if (at::hasMKL() && it->is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(it->dtype(), "GeluKernelImpl", [&]() {
      GeluKernelMKLImpl<scalar_t>(it);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(it->dtype(), "GeluKernelImpl", [&]() {
      GeluKernelImplInternal<scalar_t>(it);
    });
  }
}

#if AT_MKL_ENABLED()

template <typename T>
void GeluBackwardKernelMKLImpl(const Tensor& dY, const Tensor& X, Tensor* dX);

// TODO(yangxm): Implement this by using template functions.
#define DELEGATE_GELU_BACKWARD_KERNEL_MKL_IMPL(T, CdfNormFunc, ExpFunc)     \
  template <>                                                               \
  void GeluBackwardKernelMKLImpl<T>(                                        \
      const Tensor& dY, const Tensor& X, Tensor* dX) {                      \
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);                   \
    Tensor scratch = at::native::empty_like(X);                             \
    const int64_t N = X.numel();                                            \
    const T* dY_data = dY.data<T>();                                        \
    const T* X_data = X.data<T>();                                          \
    T* dX_data = dX->data<T>();                                             \
    T* scratch_data = scratch.data<T>();                                    \
    CdfNormFunc(N, X_data, scratch_data);                                   \
    for (int64_t i = 0; i < N; ++i) {                                       \
      dX_data[i] = -T(0.5) * X_data[i] * X_data[i];                         \
    }                                                                       \
    ExpFunc(N, dX_data, dX_data);                                           \
    for (int64_t i = 0; i < N; ++i) {                                       \
      dX_data[i] =                                                          \
          dY_data[i] * (scratch_data[i] + X_data[i] * dX_data[i] * kAlpha); \
    }                                                                       \
  }
DELEGATE_GELU_BACKWARD_KERNEL_MKL_IMPL(float, vsCdfNorm, vsExp)
DELEGATE_GELU_BACKWARD_KERNEL_MKL_IMPL(double, vdCdfNorm, vdExp)
#undef DELEGATE_GELU_BACKWARD_KERNEL_MKL_IMPL

#else // AT_MKL_ENABLED()

template <typename T>
void GeluBackwardKernelMKLImpl(const Tensor& dY, const Tensor& X, Tensor* dX) {
  AT_ASSERTM(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED()

template <typename T>
void GeluBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    Tensor* dX) {
  constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
  Tensor scratch = at::native::empty_like(X);
  const int64_t N = X.numel();
  const T* dY_data = dY.data<T>();
  const T* X_data = X.data<T>();
  T* dX_data = dX->data<T>();
  T* scratch_data = scratch.data<T>();
  for (int64_t i = 0; i < N; ++i) {
    scratch_data[i] = X_data[i] * M_SQRT1_2;
    dX_data[i] = -T(0.5) * X_data[i] * X_data[i];
  }
  // TODO(yangxm): Consider let forward pass preserve CdfNorm(X) in training
  // pass to reduce this extra tensor.
  scratch.erf_();
  dX->exp_();
  for (int64_t i = 0; i < N; ++i) {
    dX_data[i] = dY_data[i] *
        (T(0.5) * (T(1) + scratch_data[i]) + X_data[i] * dX_data[i] * kAlpha);
  }
}

void GeluBackwardKernelImpl(const Tensor& dY, const Tensor& X, Tensor* dX) {
  if (hasMKL()) {
    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "GeluBackwardKernelImpl", [&]() {
          GeluBackwardKernelMKLImpl<scalar_t>(dY, X, dX);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "GeluBackwardKernelImpl", [&]() {
          GeluBackwardKernelImplInternal<scalar_t>(dY, X, dX);
        });
  }
}

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
REGISTER_DISPATCH(GeluKernel, &GeluKernelImpl);
REGISTER_DISPATCH(GeluBackwardKernel, &GeluBackwardKernelImpl);

} // namespace native
} // namespace at
