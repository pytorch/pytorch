#pragma once

#include <zendnn.h>
#include <zendnn.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#endif

namespace zendnn {
namespace utils {

template <
    typename F,
    typename T,
    typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T>& vec, const F& f) {
  std::vector<U> result;
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
  return result;
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) {
  return val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

template <typename T>
inline bool any_le(const std::vector<T>& v, T i) {
  return std::any_of(v.begin(), v.end(), [i](T k) { return k <= i; });
}

inline memory::dims get_compatible_dilates(
    const memory::dims& dilates,
    int input_size = 4) {
  if (!dilates.empty() && !any_le(dilates, static_cast<dim>(0)))
    return fmap(dilates, [](dim x) { return x - 1; });
  if (4 == input_size) {
    return {0, 0};
  } else {
    return {0, 0, 0};
  }
}

inline memory::dims group_dims(const dims& adims, dim groups) {
  auto new_dims = adims;
  new_dims.insert(new_dims.begin(), groups);
  new_dims[1] /= groups;
  return new_dims;
}

inline std::pair<std::vector<float>, std::vector<float>> compute_scales(
    float src_scale,
    float dst_scale,
    std::vector<float> weight_scales) {
  auto scale_size = weight_scales.size();
  std::vector<float> bias_scales(scale_size), op_scales(scale_size);

  for (int i = 0; i < scale_size; i++) {
    bias_scales[i] = src_scale * weight_scales[i];
    op_scales[i] = dst_scale / bias_scales[i];
  }
  return std::make_pair(std::move(bias_scales), std::move(op_scales));
}

using bytestring = std::string;

/* Definitions for builtins unavailable on MSVC */
// see
// https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/builtins/int_lib.h
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
uint32_t __inline clz(uint32_t x) {
  unsigned long leading_zero = 0;
  if (_BitScanReverse(&leading_zero, x))
    return 31 - leading_zero;
  return 32;
}
#else
uint32_t __inline clz(uint32_t x) {
  return __builtin_clz(x);
}
#endif

inline void to_bytes(bytestring& bytes, const int arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  if (arg == 0)
    return;
  auto len = sizeof(arg) - (clz(arg) / 8);
  bytes.append(as_cstring, len);
}

inline void to_bytes(bytestring& bytes, const bool arg) {
  to_bytes(bytes, arg ? 1 : 0);
  bytes.append(1, 'b');
}

inline void to_bytes(bytestring& bytes, const float arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(float));
}

inline void to_bytes(bytestring& bytes, const uint64_t arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(uint64_t));
}

inline void to_bytes(bytestring& bytes, const int64_t arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(int64_t));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>& arg) {
  if (arg.size() > 0) {
    for (T elems : arg) {
      to_bytes(bytes, elems);
      bytes.append(1, 'v');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'v');
  }
}

template <typename T>
inline void to_bytes(bytestring& bytes, const std::vector<T>& arg) {
  // remove constness, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, const_cast<std::vector<T>&>(arg));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>&& arg) {
  // `arg` is an lval ref now, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, arg);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_enum<T>::value>::type>
inline void to_bytes(bytestring& bytes, T arg) {
  to_bytes(bytes, static_cast<int>(arg));
}

template <
    typename T,
    typename = typename std::enable_if<std::is_class<T>::value>::type,
    typename = void>
inline void to_bytes(bytestring& bytes, const T arg) {
  arg.to_bytes(bytes);
}

template <typename T, typename... Ts>
inline void to_bytes(bytestring& bytes, T&& arg, Ts&&... args) {
  to_bytes(bytes, std::forward<T>(arg));
  bytes.append(1, '*');
  to_bytes(bytes, std::forward<Ts>(args)...);
}

/** sorts an array of values using @p comparator. While sorting the array
 * of value, the function permutes an array of @p keys accordingly.
 *
 * @note The arrays of @p keys can be omitted. In this case the function
 *       sorts the array of @vals only.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(T* vals, U* keys, size_t size, F comparator) {
  if (size == 0)
    return;

  for (auto i = 0; i < size - 1; ++i) {
    bool swapped = false;
    for (auto j = 0; j < size - i - 1; j++) {
      if (comparator(vals[j], vals[j + 1]) > 0) {
        std::swap(vals[j], vals[j + 1]);
        if (keys)
          std::swap(keys[j], keys[j + 1]);
        swapped = true;
      }
    }

    if (swapped == false)
      break;
  }
}

template <typename T>
inline T rnd_up(const T a, const T b) {
  return (a + b - 1) / b * b;
}

inline int op_scale_mask(dim scale_size) {
  return scale_size > 1 ? 2 : 0;
}

inline int tensor_scale_mask(dim scale_size, bool grouped) {
  return scale_size > 1 ? grouped ? 3 : 1 : 0;
}

inline int tensor_zp_mask(dim zp_size) {
  return zp_size > 1 ? 1 : 0;
}

inline uintptr_t mod_ptr(void* ptr, size_t bytes) {
  return reinterpret_cast<uintptr_t>(ptr) & (bytes - 1);
}

inline bool is_aligned_ptr(void* ptr, size_t bytes) {
  return mod_ptr(ptr, bytes) == 0;
}

template <typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
  for (auto i = 0; i < size; ++i)
    dst[i] = src[i];
}

template <typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
  for (auto i = 0; i < size; ++i)
    if (a1[i] != a2[i])
      return false;
  return true;
}

template <typename T, typename U>
inline void array_set(T* arr, const U& val, size_t size) {
  for (auto i = 0; i < size; ++i)
    arr[i] = static_cast<T>(val);
}

} // namespace utils
} // namespace zendnn
