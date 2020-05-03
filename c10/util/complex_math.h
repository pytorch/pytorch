namespace std {

// Exponential functions

#if CUDA_VERSION < 10000
#define CUDA92_BUG(x) thrust::complex<T>(x.real(), x.imag())
#else
#define CUDA92_BUG(x) x
#endif

template<typename T>
C10_HOST_DEVICE c10::complex<T> exp(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::exp(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> log(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::log(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> log10(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log10(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::log10(static_cast<std::complex<T>>(x)));
#endif
}

// Power functions

template<typename T>
C10_HOST_DEVICE c10::complex<T> sqrt(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sqrt(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sqrt(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> pow(c10::complex<T> x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> pow(c10::complex<T> x, const T &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> pow(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE c10::complex<decltype(T() * U())> pow(c10::complex<T> x, c10::complex<U> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE c10::complex<decltype(T() * U())> pow(c10::complex<T> x, const U &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(CUDA92_BUG(x)), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T, typename U>
C10_HOST_DEVICE c10::complex<decltype(T() * U())> pow(const T &x, c10::complex<U> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(CUDA92_BUG(y))));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template<typename T>
C10_HOST_DEVICE c10::complex<T> sin(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sin(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> cos(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cos(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> tan(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tan(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> asin(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asin(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> acos(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acos(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::acos(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> atan(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atan(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template<typename T>
C10_HOST_DEVICE c10::complex<T> sinh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sinh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> cosh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cosh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> tanh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tanh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> asinh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asinh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> acosh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acosh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
C10_HOST_DEVICE c10::complex<T> atanh(const c10::complex<T> &x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atanh(static_cast<thrust::complex<T>>(CUDA92_BUG(x))));
#else
  return static_cast<c10::complex<T>>(std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

#undef CUDA92_BUG

} // namespace std
