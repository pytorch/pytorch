namespace std {

// Exponential functions

#if defined(CUDA_VERSION) && (CUDA_VERSION < 10000)
#define CUDA92_BUG(x) thrust::complex<T>(x.real(), x.imag())
#else
#define CUDA92_BUG(x) x
#endif

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> exp(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::exp(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> log(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::log(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> log10(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log10(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::log10(static_cast<std::complex<T>>(x)));
#endif
}

// Power functions

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> sqrt(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sqrt(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sqrt(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(const c10::complex<T> &x, const c10::complex<T> &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(const c10::complex<T> &x, const T &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(const T &x, const c10::complex<T> &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(const c10::complex<T> &x, const c10::complex<U> &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(const c10::complex<T> &x, const U &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(const T &x, const c10::complex<U> &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> sin(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sin(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> cos(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cos(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> tan(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tan(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> asin(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asin(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> acos(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acos(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::acos(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> atan(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atan(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> sinh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sinh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> cosh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cosh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> tanh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tanh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> asinh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asinh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> acosh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acosh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> atanh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atanh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

// We also support some math functions that is not supported by standard library:

template<typename T>
C10_HOST_DEVICE inline c10::complex<T> log1p(const c10::complex<T> &z) {
  // log1p(z) = log(1+z)
  // Let's define 1 + z = r*e^(i*a), then we have
  // log(r*e^(i*a)) = log(r) + i*a
  // but log(r) could have precision issue when |z| << 1, so we should really
  // be using log1p(r-1), where the r-1 should be computed in high precision.
  // to do so, we are doing the following transformation: (assuming z = x+iy)
  // r-1 = (r-1)*(r+1)/(r+1) = (r^2-1) / (r+1)
  //     = ((x+1)^2 + y^2 - 1) / (r+1)
  //     = (x^2 + y^2 + 2x) / (r+1)
  T x = z.real();
  T y = z.imag();
  c10::complex<T> p1 = z + T(1);
  T r = std::abs(p1);
  T a = std::arg(p1);
  T rm1 = (x * x + y * y + x * T(2)) / (r + 1);
  return {std::log1p(rm1), a};
}

#undef CUDA92_BUG

} // namespace std
