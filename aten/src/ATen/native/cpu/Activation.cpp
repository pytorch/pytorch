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

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
REGISTER_DISPATCH(GeluKernel, &GeluKernelImpl);

} // namespace native
} // namespace at
