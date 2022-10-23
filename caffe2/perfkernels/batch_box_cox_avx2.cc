#ifdef CAFFE2_PERF_USE_MKL
#include <c10/util/irange.h>
#include <caffe2/perfkernels/common.h>
#include <folly/SingletonThreadLocal.h>

#include <cstdint>
#include <cmath>
#include <vector>

#include <mkl.h>

namespace caffe2::details {

// MKL VML function templates.
template <typename T>
void PackV(const int N, const T* a, const int* ia, T* y);
template <typename T>
void UnpackV(const int N, const T* a, T* y, const int* iy);
template <typename T>
void Pow(const int N, const T* a, const T* b, T* y);
template <typename T>
void Add(const int N, const T* a, const T* b, T* y);
template <typename T>
void Div(const int N, const T* a, const T* b, T* y);
template <typename T>
void Ln(const int N, const T* a, T* y);

#define DELEGATE_PACKV_FUNCTION(T, OriginalFunc)                \
  template <>                                                   \
  void PackV<T>(const int N, const T* a, const int* ia, T* y) { \
    OriginalFunc(N, a, ia, y);                                  \
  }
DELEGATE_PACKV_FUNCTION(float, vsPackV)
DELEGATE_PACKV_FUNCTION(double, vdPackV)
#undef DELEGATE_PACKV_FUNCTION

#define DELEGATE_UNPACKV_FUNCTION(T, OriginalFunc)                \
  template <>                                                     \
  void UnpackV<T>(const int N, const T* a, T* y, const int* iy) { \
    OriginalFunc(N, a, y, iy);                                    \
  }
DELEGATE_UNPACKV_FUNCTION(float, vsUnpackV)
DELEGATE_UNPACKV_FUNCTION(double, vdUnpackV)
#undef DELEGATE_UNPACKV_FUNCTION

#define DELEGATE_SIMPLE_BINARY_FUNCTION(T, Funcname, OriginalFunc) \
  template <>                                                      \
  void Funcname<T>(const int N, const T* a, const T* b, T* y) {    \
    OriginalFunc(N, a, b, y);                                      \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Pow, vsPow)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Pow, vdPow)
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Add, vsAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Add, vdAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Div, vsDiv)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Div, vdDiv)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, OriginalFunc) \
  template <>                                                     \
  void Funcname<T>(const int N, const T* a, T* y) {               \
    OriginalFunc(N, a, y);                                        \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Ln, vsLn)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Ln, vdLn)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

template <typename T>
void box_cox_zero_lambda(
    size_t D,
    const T* const self_data,
    const T* const lambda2_data,
    T k_eps,
    T* const output_data) {
  Add(D, self_data, lambda2_data, output_data);
  for (const auto j : c10::irange(D)) {
    output_data[j] = std::max(output_data[j], k_eps);
  }

  Ln(D, output_data, output_data);
}

template <typename T>
void box_cox_nonzero_lambda(
    size_t D,
    const T* const self_data,
    const T* const lambda1_data,
    const T* const lambda2_data,
    T k_eps,
    T* const output_data) {
  Add(D, self_data, lambda2_data, output_data);
  for (const auto j : c10::irange(D)) {
    output_data[j] = std::max(output_data[j], k_eps);
  }

  // output = output ^ lambda1
  Pow(D, output_data, lambda1_data, output_data);
  // output = (output  - 1)/ lambda1
  for (const auto j : c10::irange(D)) {
    output_data[j] -= 1.0;
  }
  Div(D, output_data, lambda1_data, output_data);
}

template <typename T>
void box_cox_mixed_lambda(
    const T* const self_data,
    const std::vector<int>& nonzeros,
    const std::vector<int>& zeros,
    const T* const lambda1,
    const T* const lambda2,
    const T* const lambda2_z_,
    T k_eps,
    T* const buffer,
    T* const output_data) {
  PackV(nonzeros.size(), self_data, nonzeros.data(), buffer);
  box_cox_nonzero_lambda<T>(
      nonzeros.size(), buffer, lambda1, lambda2, k_eps, buffer);
  UnpackV(nonzeros.size(), buffer, output_data, nonzeros.data());

  PackV(zeros.size(), self_data, zeros.data(), buffer);
  box_cox_zero_lambda<T>(
      zeros.size(), buffer, lambda2_z_, k_eps, buffer);
  UnpackV(zeros.size(), buffer, output_data, zeros.data());
}

template <typename T>
void TileArrayIntoVector(
    const T* const a,
    const size_t D,
    const int K,
    std::vector<T>& b) {
  b.resize(K * D);
  for (const auto k : c10::irange(K)) {
    std::copy(a, a + D, b.begin() + k * D);
  }
}

void TileIndicesInPlace(std::vector<int>& v, const std::size_t D, const std::size_t K) {
  auto n = v.size();
  v.resize(K * n);
  for (const auto k : c10::irange(1, K)) {
    for (const auto j : c10::irange(n)) {
      v[k * n + j] = v[j] + k * D;
    }
  }
}

template <typename T>
void compute_batch_box_cox__avx2_fma(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const T* self_data,
    const T* __restrict lambda1_data,
    const T* __restrict lambda2_data,
    T* output_data) {
  constexpr T k_eps = static_cast<T>(1e-6);

  FOLLY_DECLARE_REUSED(zeros, std::vector<int>);
  FOLLY_DECLARE_REUSED(nonzeros, std::vector<int>);
  // Don't bother calling reserve; calls after the first will get a
  // correctly-sized allocation anyway.
  for (const auto j : c10::irange(D)) {
    if (lambda1_data[j] == 0) {
      zeros.push_back(j);
    } else {
      nonzeros.push_back(j);
    }
  }

  // Process K rows at a time for effective vectorization with small rows.
  const auto K = std::min(N, (block_size + D - 1) / D);

  FOLLY_DECLARE_REUSED(lambda1_, std::vector<T>);
  FOLLY_DECLARE_REUSED(lambda2_, std::vector<T>);
  FOLLY_DECLARE_REUSED(lambda2_z_, std::vector<T>);

  if (nonzeros.size() == D) {
    // ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0
    size_t i = 0;
    if (K > 1) {
      TileArrayIntoVector(lambda1_data, D, K, lambda1_);
      TileArrayIntoVector(lambda2_data, D, K, lambda2_);
      DCHECK_EQ(K * D, lambda1_.size());
      DCHECK_EQ(K * D, lambda2_.size());
      for (; i < N - K + 1; i += K, self_data += K * D, output_data += K * D) {
        box_cox_nonzero_lambda<T>(
            K * D,
            self_data,
            lambda1_.data(),
            lambda2_.data(),
            k_eps,
            output_data);
      }
    }
    for (; i < N; i++, self_data += D, output_data += D) {
      box_cox_nonzero_lambda<T>(
          D, self_data, lambda1_data, lambda2_data, k_eps, output_data);
    }
  } else if (zeros.size() == D) {
    // ln(x + lambda2), if lambda1 == 0
    size_t i = 0;
    if (K > 1) {
      TileArrayIntoVector(lambda2_data, D, K, lambda2_z_);
      DCHECK_EQ(K * D, lambda2_z_.size());
      for (; i < N - K + 1; i += K, self_data += K * D, output_data += K * D) {
        box_cox_zero_lambda<T>(
            K * D, self_data, lambda2_z_.data(), k_eps, output_data);
      }
    }
    for (; i < N; i++, self_data += D, output_data += D) {
      box_cox_zero_lambda<T>(
          D, self_data, lambda2_data, k_eps, output_data);
    }
  } else {
    // mix zeros and nonzeros
    const size_t n = nonzeros.size();
    if (K > 1) {
      TileIndicesInPlace(nonzeros, 0, K);
      TileIndicesInPlace(zeros, 0, K);
    }

    FOLLY_DECLARE_REUSED(buffer, std::vector<T>);

    buffer.resize(std::max(nonzeros.size(), zeros.size()));
    lambda1_.resize(nonzeros.size());
    lambda2_.resize(nonzeros.size());
    lambda2_z_.resize(zeros.size());
    PackV(nonzeros.size(), lambda1_data, nonzeros.data(), lambda1_.data());
    PackV(nonzeros.size(), lambda2_data, nonzeros.data(), lambda2_.data());
    PackV(zeros.size(), lambda2_data, zeros.data(), lambda2_z_.data());

    size_t i = 0;
    if (K > 1) {
      // Truncate to original size, and re-tile with offsets this time.
      nonzeros.resize(n);
      DCHECK_GT(D, n);
      zeros.resize(D - n);
      TileIndicesInPlace(nonzeros, D, K);
      TileIndicesInPlace(zeros, D, K);
      DCHECK_EQ(nonzeros.size(), lambda1_.size());
      DCHECK_EQ(nonzeros.size(), lambda2_.size());
      DCHECK_EQ(zeros.size(), lambda2_z_.size());

      for (; i < N - K + 1; i += K, self_data += K * D, output_data += K * D) {
        box_cox_mixed_lambda<T>(
            self_data,
            nonzeros,
            zeros,
            lambda1_.data(),
            lambda2_.data(),
            lambda2_z_.data(),
            k_eps,
            buffer.data(),
            output_data);
      }
      // Truncate to original size.
      nonzeros.resize(n);
      zeros.resize(D - n);
    }
    for (; i < N; i++, self_data += D, output_data += D) {
      box_cox_mixed_lambda<T>(
          self_data,
          nonzeros,
          zeros,
          lambda1_.data(),
          lambda2_.data(),
          lambda2_z_.data(),
          k_eps,
          buffer.data(),
          output_data);
    }
  }
};


template
void compute_batch_box_cox__avx2_fma<float>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const float* self_data,
  const float* __restrict lambda1_data,
  const float* __restrict lambda2_data,
  float* output_data);

template
void compute_batch_box_cox__avx2_fma<double>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const double* self_data,
  const double* __restrict lambda1_data,
  const double* __restrict lambda2_data,
  double* output_data);

} // namespace caffe2::detail
#endif
