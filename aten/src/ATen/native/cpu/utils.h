#pragma once

#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/llvmMathExtras.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif

namespace at {
namespace native {

inline namespace CPU_CAPABILITY {

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

// Helper struct for bfloat16 vectorization
// Useful when you need float as immediate dtype or accumulate dtype
using namespace vec;
struct Vec2 {
  Vectorized<float> val0, val1;
  Vec2(Vectorized<float> v0, Vectorized<float> v1) : val0(v0), val1(v1) {}
  Vec2(float v) : val0(v), val1(v) {}
  static Vec2 loadu(const BFloat16* ptr) {
    Vectorized<float> v0, v1;
    std::tie(v0, v1) = convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr));
    return {v0, v1};
  }
  static Vec2 loadu(const float* ptr) {
    return {Vectorized<float>::loadu(ptr), Vectorized<float>::loadu(ptr + Vectorized<float>::size())};
  }
  void store(BFloat16* ptr) const {
    Vectorized<BFloat16> val = convert_float_bfloat16(val0, val1);
    val.store(ptr);
  }
  void store(float* ptr) const {
    val0.store(ptr);
    val1.store(ptr + Vectorized<float>::size());
  }
};
inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.val0 + b.val0, a.val1 + b.val1}; }
inline Vec2 operator*(const Vec2& a, const Vec2& b) { return {a.val0 * b.val0, a.val1 * b.val1}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.val0 - b.val0, a.val1 - b.val1}; }
inline Vec2 operator/(const Vec2& a, const Vec2& b) { return {a.val0 / b.val0, a.val1 / b.val1}; }
inline Vec2 maximum(const Vec2& a, const Vec2& b) { return {vec::maximum(a.val0, b.val0), vec::maximum(a.val1, b.val1)}; }
inline Vec2 minimum(const Vec2& a, const Vec2& b) { return {vec::minimum(a.val0, b.val0), vec::minimum(a.val1, b.val1)}; }

template <typename scalar_t> struct VectorizedType { using type = Vectorized<scalar_t>; };
template <> struct VectorizedType<BFloat16> { using type = Vec2; };
template <typename scalar_t> using VecType = typename VectorizedType<scalar_t>::type;

// Helper for mixed data type parameter Vec::load
inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const BFloat16* ptr) {
  return convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr));
}

inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const float* ptr) {
  using Vec = Vectorized<float>;
  return std::make_tuple(Vec::loadu(ptr), Vec::loadu(ptr + Vec::size()));
}

inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const BFloat16* ptr, int64_t count) {
  return convert_bfloat16_float(Vectorized<BFloat16>::loadu(ptr, count));
}

inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const float* ptr, int64_t count) {
  using Vec = Vectorized<float>;
  if (count > Vec::size()) {
  return std::make_tuple(Vec::loadu(ptr), Vec::loadu(ptr + Vec::size(), count - Vec::size()));
  } else {
    return std::make_tuple(Vec::loadu(ptr, count), Vec(0));
  }
}

} // namespace

namespace utils {

template <typename T>
T CeilLog2(const T& x) {
  if (x <= 2) {
    return 1;
  }
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<T>(llvm::findLastSet(static_cast<uint64_t>(x) - 1)) + 1;
}

// matrix transpose:
//   src has shape of M by N, with leading dimension of ld_src
//   dst has shape of N by M, with leading dimension of ld_dst
template <typename T>
inline void transpose(int64_t M, int64_t N, const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (int64_t j = 0; j < N; j++) {
    for (int64_t i = 0; i < M; i++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

#ifdef USE_FBGEMM
template <>
inline void transpose<float>(int64_t M, int64_t N, const float* src, int64_t ld_src, float* dst, int64_t ld_dst) {
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  fbgemm::transpose_simd<float>(M, N, src, ld_src, dst, ld_dst);
}
#endif

template <typename index_t, typename F>
inline void parallel_sparse_csr(
    const TensorAccessor<index_t, 1>& crow_acc,
    const int64_t M,
    const int64_t nnz,
    const F& f) {
  TORCH_CHECK(crow_acc.size(0) == M + 1);

  // directly parallel on `M` may lead to load imbalance,
  // statically determine thread partition here to average payload
  // for each thread.
  int num_threads = at::get_num_threads();
  std::vector<int64_t> thread_splits(num_threads + 1, M);

  int64_t thread_averge_payload = std::max((int64_t)1, divup(nnz, num_threads));

  thread_splits[0] = 0;
  int64_t sum = 0;
  int64_t t = 1;
  for (const auto m : c10::irange(M)) {
    int64_t row_start = crow_acc[m];
    int64_t row_end = crow_acc[m + 1];
    sum += row_end - row_start;
    if (sum > t * thread_averge_payload) {
      thread_splits[t] = m;
      t++;
    }
  }
  // need to restore the last index,
  // due to rounding error when calculating `thread_averge_payload`.
  thread_splits[num_threads] = M;

  at::parallel_for(0, num_threads, 1, [&](int64_t cbegin, int64_t cend) {
    int tid = at::get_thread_num();
    int64_t begin = thread_splits[tid];
    int64_t end = thread_splits[tid + 1];
    f(begin, end);
  });
}

} // namespace utils

} // namespace native
} // namespace at
