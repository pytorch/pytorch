#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <omp.h>

// WARNING: be extra careful when including more ATen/c10 header files here!
// Because AOTInductor generated code will copy-paste this cpp_prefix.h for
// the CPU backend, we have to make sure the used headers are implemented
// in a header-only way, i.e. all the function and class definitions are
// in .h files instead of .cpp files, to avoid ABI backward-compatiblity breakage.

#include <ATen/NumericUtils.h>
#include <ATen/core/PhiloxRNGEngine.h>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/generic_math.h>
#include <c10/util/Half.h>
#include <c10/util/TypeCast.h>

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON)
#define INDUCTOR_USE_VECTOR_TYPES() 1
#else
#define INDUCTOR_USE_VECTOR_TYPES() 0
#endif

#if INDUCTOR_USE_VECTOR_TYPES()
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#else
// For calc_erfinv
#include <ATen/native/Math.h>
#endif

typedef at::Half half;
typedef at::BFloat16 bfloat16;

typedef at::Float8_e4m3fn float8_e4m3fn;
typedef at::Float8_e5m2 float8_e5m2;

template <typename T>
struct Welford {
  T mean = T(0);
  T m2 = T(0);
  int64_t index = 0;
};


template <typename T>
struct IsVecType: std::false_type {};

#if INDUCTOR_USE_VECTOR_TYPES()
template <typename T>
struct IsVecType<at::vec::Vectorized<T>>: std::true_type {};
#endif

template <typename T>
struct WeightRecp {
  using scalar_t = typename T::value_type;
  int64_t N;
  std::vector<scalar_t> weight_recps;
  WeightRecp(int64_t N) : N(N) {
    weight_recps.reserve(N);
    for (const auto i : c10::irange(N)) {
      weight_recps.push_back(
          scalar_t(static_cast<double>(1) / static_cast<double>(i + 1)));
    }
  }
};

template <typename T>
Welford<T> welford_combine(const Welford<T> &a, const Welford<T> &b) {
  if (a.index == 0) {
    return b;
  }
  if (b.index == 0) {
    return a;
  }
  auto delta = b.mean - a.mean;
  auto new_index = a.index + b.index;
  auto wb_over_w = T(b.index) / T(new_index);
  auto result = Welford<T>{
    a.mean + delta * wb_over_w,
    a.m2 + b.m2 + delta * delta * T(a.index) * wb_over_w,
    new_index,
  };
  return result;
}

template <typename T>
Welford<T> welford_combine(const Welford<T> &acc, T data, const WeightRecp<T>* w=nullptr) {
  // Add a single data point
  int64_t index = acc.index + 1;
  auto delta = data - acc.mean;
  T new_mean;
  if constexpr (!IsVecType<T>::value) {
    new_mean = acc.mean + delta / T(index);
  } else {
    new_mean = acc.mean +
      ((w == nullptr || acc.index >= w->weight_recps.size())
            ? delta / T(index)
            : delta * T(w->weight_recps[acc.index]));
  }
  auto new_delta = data - new_mean;
  auto result = Welford<T>{
    new_mean,
    acc.m2 + delta * new_delta,
    index
  };
  return result;
}

// Refer to https://github.com/pytorch/pytorch/blob/b5b36cf0c4e1958f1ff25120f5d4beeef3288187/
// aten/src/ATen/native/SharedReduceOps.h#L419-L445
template <typename scalar_t>
inline bool greater_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
  // If (a == b), then choose the one with lower idx, else max(a, b)
  if (at::_isnan(a)) {
    if (at::_isnan(b)) {
      return idx_a < idx_b;
    }
    return true;
  }
  return (a == b) ? idx_a < idx_b : (a > b);
}

template <typename scalar_t>
inline bool less_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
  // If (a == b), then choose the one with lower idx, else min(a, b)
  if (at::_isnan(a)) {
    if (at::_isnan(b)) {
      return idx_a < idx_b;
    }
    return true;
  }
  return (a == b) ? idx_a < idx_b : (a < b);
}

#if INDUCTOR_USE_VECTOR_TYPES()
template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> vec_shuffle_down(at::vec::Vectorized<scalar_t> x, size_t n) {
  using Vec = at::vec::Vectorized<scalar_t>;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  x.store(array);
  for (size_t i = 0; i + n < Vec::size(); i += 2 * n) {
    array[i] = array[i + n];
  }
  return Vec::loadu(array);
}

#ifdef CPU_CAPABILITY_AVX2
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
  case 1:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
  case 2:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
  case 4:
    return vec_t(_mm256_permute2f128_ps(x, x, SHUFFLE_MASK(1, 1, 1, 1)));
  }
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

#ifdef CPU_CAPABILITY_AVX512
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
    case 1:
      return vec_t(_mm512_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
    case 2:
      return vec_t(_mm512_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
    case 4:
      return vec_t(_mm512_permutexvar_ps(
          _mm512_set_epi32(
              12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4),
          x));
    case 8:
      return vec_t(_mm512_permutexvar_ps(
          _mm512_set_epi32(8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8), x));
  }
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<at::vec::Vectorized<scalar_t>> acc) {
  using Vec = at::vec::Vectorized<scalar_t>;
  for (size_t n = 1; n < Vec::size(); n *= 2) {
    auto index = acc.index;
    auto shuffled = Welford<Vec>{
      vec_shuffle_down(acc.mean, n),
      vec_shuffle_down(acc.m2, n),
      index,
    };
    acc = welford_combine(acc, shuffled);
  }

  Welford<scalar_t> result;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  acc.mean.store(array);
  result.mean = array[0];

  acc.m2.store(array);
  result.m2 = array[0];

  result.index = acc.index;

  return result;
}
#endif


template <typename T, typename U> inline typename std::common_type<T, U>::type mod(T a, U b) { return a % b; }
template <> inline float mod(float a, float b) { return std::fmod(a, b); }
template <> inline double mod(double a, double b) { return std::fmod(a, b); }

template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a < b ? a : b;
}

constexpr float uint32_to_uniform_float(uint32_t value) {
  // maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
  constexpr float scale = 4.6566127342e-10;
  return static_cast<float>(value & 0x7FFFFFFF) * scale;
}

float normalized_rand_cpu(uint32_t seed, uint32_t offset) {
  return uint32_to_uniform_float(at::Philox4_32(seed, 0, offset)());
}

float randn_cpu(uint32_t seed, uint32_t offset) {
  at::Philox4_32 engine(seed, 0, offset);
  return engine.randn(10);
}

int64_t randint64_cpu(uint32_t seed, uint32_t offset, int64_t low, int64_t high) {
  auto gen = at::Philox4_32(seed, 0, offset);
  uint64_t r0 = gen();
  uint64_t r1 = gen();
  uint64_t result = r0 | (r1 << 32);
  return static_cast<int64_t>(result % (high - low)) + low;
}

template <typename T> struct AsIntegerType { typedef T type; };
template <> struct AsIntegerType<float> { typedef uint32_t type; };
template <> struct AsIntegerType<double> { typedef uint64_t type; };
template <> struct AsIntegerType<bfloat16> { typedef uint16_t type; };

template <typename T>
typename std::enable_if_t<!std::is_reduced_floating_point_v<T>, T>
inline fetch_value(volatile T *addr) {
  return *addr;
}

template <typename T>
typename std::enable_if_t<std::is_reduced_floating_point_v<T>, T>
inline fetch_value(volatile T *addr) {
  return T(addr->x, T::from_bits());
}

template <typename T>
typename std::enable_if_t<!std::is_integral_v<T>>
atomic_add(volatile T *addr, T offset) {
  typedef typename AsIntegerType<T>::type alt_type;

  static_assert(sizeof(std::atomic<alt_type>) == sizeof(T),
                "std::atomic issue");

  alt_type expected;

  alt_type desired;

  std::atomic<alt_type> *atomic_addr = (std::atomic<alt_type> *)addr;
  do {
    T val = fetch_value(addr);
    reinterpret_cast<T *>(&expected)[0] = val;
    reinterpret_cast<T *>(&desired)[0] = val + offset;
  } while (!atomic_addr->compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed));
}

// Since C++20 float is supported by fetch_add, but the performance may not
// better than compare_exchange_weak, which can be checked by microbenchmark
// inductor_cpu_atomic.py
template <typename T>
typename std::enable_if_t<std::is_integral_v<T>>
atomic_add(volatile T *addr, T offset) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T),
                "std::atomic issue");
  std::atomic<T> *atomic_addr = (std::atomic<T> *)addr;
  atomic_addr->fetch_add(offset, std::memory_order_relaxed);
}

void mm_get_thread_blocking(
    int num_threads,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t M0,
    int64_t N0,
    int64_t K0,
    int64_t& Mt,
    int64_t& Nt,
    int64_t& Kt) {
  auto get_factors = [](int64_t number) {
    int count = 0;
    for (int64_t i = std::sqrt(number); i > 0; --i) {
      if (number % i == 0) {
        count += 2;
      }
    }
    auto factors = std::make_unique<int64_t[]>(count);
    int index = 0;
    for (int64_t i = std::sqrt(number); i > 0; --i) {
      if (number % i == 0) {
        factors[index++] = number / i;
        factors[index++] = i;
      }
    }
    return std::make_tuple(std::move(factors), count);
  };

  auto get_blocking = [](int64_t num_threads,
                         int64_t factor,
                         int64_t m_blocks,
                         int64_t n_blocks,
                         int64_t k_blocks) {
    int64_t thread_block_n = (n_blocks + factor - 1) / factor;
    int64_t cofactor = num_threads / factor;
    int64_t thread_block_m = (m_blocks + cofactor - 1) / cofactor;
    return std::make_tuple(thread_block_m, thread_block_n, k_blocks);
  };

  int64_t m_blocks = (M + M0 - 1) / M0;
  int64_t n_blocks = (N + N0 - 1) / N0;
  int64_t k_blocks = (K + K0 - 1) / K0;

  auto [factors, count] = get_factors(num_threads);
  assert(count > 0);

  for (int i = 0; i < count; ++i) {
    int64_t factor = factors[i];
    if (n_blocks % factor == 0 &&
        m_blocks % (num_threads / factor) == 0) {
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
  }

  for (int i = 0; i < count; ++i) {
    int64_t factor = factors[i];
    if (n_blocks % factor == 0) {
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
    int64_t cofactor = num_threads / factor;
    if (m_blocks % cofactor == 0) {
      std::tie(Mt, Nt, Kt) = get_blocking(
          num_threads, factor, m_blocks, n_blocks, k_blocks);
      return;
    }
  }

  assert(false && "Should not reach here.");
  // Dummy return to avoid compiler warning
  return;
}

inline void mm_get_thread_blocks(
    int thread_id,
    int64_t M_blocks,
    int64_t N_blocks,
    int64_t K_blocks,
    int64_t Mt_blocks,
    int64_t Nt_blocks,
    int64_t Kt_blocks,
    int64_t& m_block_start,
    int64_t& m_block_end,
    int64_t& n_block_start,
    int64_t& n_block_end,
    int64_t& k_block_start,
    int64_t& k_block_end) {
  int64_t num_Kt = (K_blocks + Kt_blocks - 1) / Kt_blocks;
  k_block_start = (thread_id % num_Kt) * Kt_blocks;
  k_block_end = std::min(k_block_start + Kt_blocks, K_blocks);
  thread_id /= num_Kt;
  int64_t num_Nt = (N_blocks + Nt_blocks - 1) / Nt_blocks;
  n_block_start = (thread_id % num_Nt) * Nt_blocks;
  n_block_end = std::min(n_block_start + Nt_blocks, N_blocks);
  thread_id /= num_Nt;
  m_block_start = std::min(thread_id * Mt_blocks, M_blocks);
  m_block_end = std::min(m_block_start + Mt_blocks, M_blocks);
}

struct amx_tilecfg {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

class AMXState {
 private:
  amx_tilecfg tilecfg_;
  uint8_t rows_;
  uint16_t colsb_;
  uint8_t num_tile_rows_;
  uint8_t num_tile_columns_;

 public:
  AMXState() : rows_(0), colsb_(0), num_tile_rows_(0), num_tile_columns_(0) {
    memset(&tilecfg_, 0, sizeof(tilecfg_));
  }

  inline void configure(
      uint8_t rows,
      uint16_t colsb,
      uint8_t num_tile_rows,
      uint8_t num_tile_columns,
      void (*loadconfig)(const amx_tilecfg&)) {
    if (tilecfg_.palette_id == 1 && rows_ == rows && colsb_ == colsb &&
        num_tile_rows_ == num_tile_rows &&
        num_tile_columns_ == num_tile_columns) {
      return;
    }
    tilecfg_.palette_id = 1;
    rows_ = rows;
    colsb_ = colsb;
    num_tile_rows_ = num_tile_rows;
    num_tile_columns_ = num_tile_columns;
    const auto num_c_tiles = num_tile_rows * num_tile_columns;
    // For C
    for (int i = 0; i < num_c_tiles; i++) {
      tilecfg_.rows[i] = rows;
      tilecfg_.colsb[i] = 64;
    }
    // For A
    for (int i = 0; i < num_tile_rows; i++) {
      tilecfg_.rows[i + num_c_tiles] = rows;
      tilecfg_.colsb[i + num_c_tiles] = colsb;
    }
    // For B
    for (int i = 0; i < num_tile_columns; i++) {
      tilecfg_.rows[i + num_c_tiles + num_tile_rows] = colsb / 4;
      tilecfg_.colsb[i + num_c_tiles + num_tile_rows] = 64;
    }
    loadconfig(tilecfg_);
  }

  inline void release(void (*tile_release)()) {
    tilecfg_.palette_id = 0;
    tile_release();
  }
};
