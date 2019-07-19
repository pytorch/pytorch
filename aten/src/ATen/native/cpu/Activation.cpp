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
    cpu_kernel_vec(
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

// TODO(yangxm): Consider to use TensorIterator here.
template <typename T>
void GeluKernelMKLImpl(const Tensor& X, Tensor* Y);

#define DELEGATE_GELU_KERNEL_MKL_IMPL(T, CdfNormFunc, MulFunc) \
  template <>                                                  \
  void GeluKernelMKLImpl<T>(const Tensor& X, Tensor* Y) {      \
    const int64_t N = X.numel();                               \
    const T* X_data = X.data<T>();                             \
    T* Y_data = Y->data<T>();                                  \
    CdfNormFunc(N, X_data, Y_data);                            \
    MulFunc(N, X_data, Y_data, Y_data);                        \
  }
DELEGATE_GELU_KERNEL_MKL_IMPL(float, vsCdfNorm, vsMul)
DELEGATE_GELU_KERNEL_MKL_IMPL(double, vdCdfNorm, vdMul)
#undef DELEGATE_GELU_KERNEL_MKL_IMPL

#else // AT_MKL_ENABLED()

template <typename T>
void GeluKernelMKLImpl(const Tensor& X, Tensor* Y) {
  AT_ASSERTM(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED()

template <typename T>
void GeluKernelImplInternal(const Tensor& X, Tensor* Y) {
  const int64_t N = X.numel();
  const T* X_data = X.data<T>();
  T* Y_data = Y->data<T>();
  for (int64_t i = 0; i < N; ++i) {
    Y_data[i] = X_data[i] * M_SQRT1_2;
  }
  Y->erf_();
  for (int64_t i = 0; i < N; ++i) {
    Y_data[i] = (Y_data[i] + T(1)) * X_data[i] * T(0.5);
  }
}

// TODO(yangxm): Add another fast kernel using formula
// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
// and the fast tanh impl from Eigen.
void GeluKernelImpl(const Tensor& X, Tensor* Y) {
  if (at::hasMKL()) {
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GeluKernelImpl", [&]() {
      GeluKernelMKLImpl<scalar_t>(X, Y);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GeluKernelImpl", [&]() {
      GeluKernelImplInternal<scalar_t>(X, Y);
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

void hardshrink_cpu_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(iter,
      [=](scalar_t self_val) {
        return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : self_val;
      },
      [=](Vec256<scalar_t> self_val) {
        return ((self_val < -lambd_val) | (self_val > lambd_val)) & self_val;
      }
    );
  });
}

void hardshrink_backward_cpu_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(iter,
      [=](scalar_t grad_val, scalar_t self_val) {
        return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : grad_val;
      },
      [=](Vec256<scalar_t> grad_val, Vec256<scalar_t> self_val) {
        return ((self_val < -lambd_val) | (self_val > lambd_val)) & grad_val;
      }
    );
  });
}

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
REGISTER_DISPATCH(GeluKernel, &GeluKernelImpl);
REGISTER_DISPATCH(GeluBackwardKernel, &GeluBackwardKernelImpl);
REGISTER_DISPATCH(hardshrink_cpu_stub, &hardshrink_cpu_kernel);
REGISTER_DISPATCH(hardshrink_backward_cpu_stub, &hardshrink_backward_cpu_kernel);

} // namespace native
} // namespace at
