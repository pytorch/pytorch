namespace std {

// Exponential functions

template<typename T>
c10::complex<T> exp(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::exp(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
c10::complex<T> log(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(std::log(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
c10::complex<T> log10(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log10(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(std::log10(static_cast<std::complex<T>>(x)));
#endif
}

// Power functions

template<typename T>
c10::complex<T> sqrt(c10::complex<T> x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sqrt(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(std::sqrt(static_cast<std::complex<T>>(x)));
#endif
}

template<typename T>
c10::complex<T> pow(c10::complex<T> x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> pow(c10::complex<T> x, const T &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T>
c10::complex<T> pow(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
c10::complex<decltype(T() * U())> pow(c10::complex<T> x, c10::complex<U> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template<typename T, typename U>
c10::complex<decltype(T() * U())> pow(c10::complex<T> x, const U &y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<c10::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template<typename T, typename U>
c10::complex<decltype(T() * U())> pow(const T &x, c10::complex<U> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template<typename T>
c10::complex<T> sin(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sin(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::sin(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> cos(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cos(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::cos(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> tan(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tan(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::tan(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> asin(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asin(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::asin(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> acos(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acos(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::acos(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> atan(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atan(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::atan(x, static_cast<std::complex<T>>(y)));
#endif
}

// Hyperbolic functions

template<typename T>
c10::complex<T> sinh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sinh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::sinh(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> cosh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cosh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::cosh(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> tanh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tanh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::tanh(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> asinh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asinh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::asinh(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> acosh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acosh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::acosh(x, static_cast<std::complex<T>>(y)));
#endif
}

template<typename T>
c10::complex<T> atanh(const T &x, c10::complex<T> y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atanh(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::atanh(x, static_cast<std::complex<T>>(y)));
#endif
}

} // namespace std