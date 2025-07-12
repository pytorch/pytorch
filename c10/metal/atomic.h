#pragma once
#include <metal_atomic>
namespace c10 {
namespace metal {

// Atomic operations helper
template <typename T>
struct AtomicType {};
template <typename T>
using AtomicType_t = typename AtomicType<T>::type;

template <>
struct AtomicType<float> {
  using type = ::metal::atomic<float>;
  static inline void atomic_add(device type* data, long offset, float value) {
    ::metal::atomic_fetch_add_explicit(
        data + offset, value, ::metal::memory_order_relaxed);
  }
};

template <>
struct AtomicType<int> {
  using type = ::metal::atomic<int>;
  static inline void atomic_add(device type* data, long offset, int value) {
    ::metal::atomic_fetch_add_explicit(
        data + offset, value, ::metal::memory_order_relaxed);
  }
};

// As of Metal3.2 atomic operations are not supported on half-precision floats,
// so they must be simulated Using atomic compare and exchange over 32-bit
// atomic type
template <typename T>
static inline void atomic_add_helper(
    device ::metal::atomic<uint>* data,
    long offset,
    T value) {
  auto ptr = data + (offset >> 1);
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  union {
    uint i;
    T t[2];
  } val;
  do {
    val.i = old;
    val.t[offset & 1] += value;
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr,
      &old,
      val.i,
      ::metal::memory_order_relaxed,
      ::metal::memory_order_relaxed));
}

template <>
struct AtomicType<half> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, half value) {
    atomic_add_helper<half>(data, offset, value);
  }
};

#if __METAL_VERSION__ >= 310
template <>
struct AtomicType<bfloat> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, bfloat value) {
    atomic_add_helper<bfloat>(data, offset, value);
  }
};
#endif

// Metal supports atomic_store_explicit for bools, but
// sizeof(::metal::atomic_bool) is 4 Therefore it could not be used to
// atomically modify unaligned memory, so fall back to compare and exchange
// trick As accumulation over booleans are just or operation, do nothing if
// value is false
template <>
struct AtomicType<bool> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, bool value) {
    if (!value) {
      return;
    }
    auto ptr = data + (offset >> 2);
    auto old =
        ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
    union {
      uint i;
      bool t[4];
    } val;
    do {
      val.i = old;
      val.t[offset & 3] = true;
    } while (!::metal::atomic_compare_exchange_weak_explicit(
        ptr,
        &old,
        val.i,
        ::metal::memory_order_relaxed,
        ::metal::memory_order_relaxed));
  }
};

} // namespace metal
} // namespace c10
