#include <ATen/native/mps/kernels/TensorCompare.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

struct clamp_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min, const T c_max) {
    return c10::metal::min(c10::metal::max(a, b_min), c_max);
  }
};

struct clamp_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const ClampScalarParams<T> params) {
    return c10::metal::min(c10::metal::max(a, params.min), params.max);
  }
};

struct clamp_min_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min) {
    return c10::metal::max(a, b_min);
  }
};

struct clamp_max_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_max) {
    return c10::metal::min(a, b_max);
  }
};

#define REGISTER_CLAMP_SCALAR_OP(T)                   \
  typedef ClampScalarParams<T> ClampScalarParams_##T; \
  REGISTER_UNARY_ALPHA_OP(clamp_scalar, T, ClampScalarParams_##T, T);

#define REGISTER_ALL_CLAMP_OPS(T)                     \
  REGISTER_TERNARY_OP(clamp, T, T);                   \
  REGISTER_CLAMP_SCALAR_OP(T);                        \
  REGISTER_UNARY_ALPHA_OP(clamp_min_scalar, T, T, T); \
  REGISTER_UNARY_ALPHA_OP(clamp_max_scalar, T, T, T);

REGISTER_ALL_CLAMP_OPS(long);
REGISTER_ALL_CLAMP_OPS(int);
REGISTER_ALL_CLAMP_OPS(short);
REGISTER_ALL_CLAMP_OPS(uchar);
REGISTER_ALL_CLAMP_OPS(char);
REGISTER_ALL_CLAMP_OPS(bool);

REGISTER_ALL_CLAMP_OPS(float);
REGISTER_ALL_CLAMP_OPS(half);
REGISTER_ALL_CLAMP_OPS(bfloat);

struct where_functor {
  template <typename T>
  inline T operator()(const T cond, const T a, const T b) {
    return c10::metal::cast_to<bool>(cond) ? a : b;
  }
};

#define REGISTER_WHERE_OP(T)                                                  \
  template [[host_name("where_dense_" #T "_bool")]] kernel void               \
  c10::metal::ternary_dense<T, where_functor, T, bool>(                       \
      device T*, device const bool*, device const T*, device const T*, uint); \
  template [[host_name("where_strided_" #T "_bool")]] kernel void             \
  c10::metal::ternary_strided<T, where_functor, T, bool>(                     \
      device void*,                                                           \
      device const char*,                                                     \
      device const char*,                                                     \
      device const char*,                                                     \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      constant uint&,                                                         \
      uint3)

REGISTER_WHERE_OP(bool);
REGISTER_WHERE_OP(uchar);
REGISTER_WHERE_OP(char);
REGISTER_WHERE_OP(short);
REGISTER_WHERE_OP(ushort);
REGISTER_WHERE_OP(int);
REGISTER_WHERE_OP(uint);
REGISTER_WHERE_OP(long);
REGISTER_WHERE_OP(ulong);
REGISTER_WHERE_OP(half);
REGISTER_WHERE_OP(bfloat);
REGISTER_WHERE_OP(float);
REGISTER_WHERE_OP(half2);
REGISTER_WHERE_OP(float2);

struct isposinf_functor {
  template <typename T>
  inline bool operator()(const T x) {
    return x == ::metal::numeric_limits<T>::infinity();
  }
};

struct isneginf_functor {
  template <typename T>
  inline bool operator()(const T x) {
    return x == -::metal::numeric_limits<T>::infinity();
  }
};

#define REGISTER_ISINF_OPS(T)           \
  REGISTER_UNARY_OP(isposinf, T, bool); \
  REGISTER_UNARY_OP(isneginf, T, bool);

REGISTER_ISINF_OPS(float);
REGISTER_ISINF_OPS(half);
REGISTER_ISINF_OPS(bfloat);

template <typename T>
inline uint lower_bound(device const T* data, uint size, T value) {
  uint lo = 0, hi = size;
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / 2;
    if (data[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T>
inline bool contains(
    device const T* test_elements,
    uint num_test,
    T element,
    bool_constant<false>) {
  bool found = false;
  for (uint i = 0; i < num_test; ++i) {
    found |= element == test_elements[i];
  }
  return found;
}

template <typename T>
inline bool contains(
    device const T* test_elements,
    uint num_test,
    T element,
    bool_constant<true>) {
  const auto lo = lower_bound(test_elements, num_test, element);
  return lo < num_test && test_elements[lo] == element;
}

template <typename T, bool SORTED>
kernel void isin(
    device const T* elements [[buffer(0)]],
    device const T* test_elements [[buffer(1)]],
    device bool* out [[buffer(2)]],
    constant uint& num_test [[buffer(3)]],
    constant bool& invert [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const auto found = contains(
      test_elements, num_test, elements[index], bool_constant<SORTED>{});
  out[index] = found != invert;
}

template <typename T, bool SORTED>
kernel void isin_mark(
    device const T* elements [[buffer(0)]],
    device const T* test_elements [[buffer(1)]],
    device atomic_uint* found [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const auto test_element = test_elements[index];
  const uint lo =
      SORTED ? lower_bound(elements, num_elements, test_element) : 0;
  for (auto i = lo; i < num_elements; ++i) {
    if (elements[i] == test_element &&
        reinterpret_cast<device const uint*>(found)[i] == 0) {
      atomic_store_explicit(found + i, 1u, memory_order_relaxed);
    }
    if IF_CONSTEXPR (SORTED) {
      break;
    }
  }
}

template <typename T, bool SORTED>
kernel void isin_gather(
    device const T* elements [[buffer(0)]],
    device const T* lookup [[buffer(1)]],
    device bool* out [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    constant bool& invert [[buffer(4)]],
    device const uint* found [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const auto element = elements[index];
  const auto i = SORTED ? lower_bound(lookup, num_elements, element) : index;
  out[index] = (i < num_elements && lookup[i] == element && found[i]) != invert;
}

#define REGISTER_ISIN_OP(NAME, T, SORTED)                      \
  template [[host_name(NAME #T)]] kernel void isin<T, SORTED>( \
      device const T*,                                         \
      device const T*,                                         \
      device bool*,                                            \
      constant uint&,                                          \
      constant bool&,                                          \
      uint)

#define REGISTER_ISIN_MARK_OP(NAME, T, SORTED)                      \
  template [[host_name(NAME #T)]] kernel void isin_mark<T, SORTED>( \
      device const T*,                                              \
      device const T*,                                              \
      device atomic_uint*,                                          \
      constant uint&,                                               \
      uint)

#define REGISTER_ISIN_GATHER_OP(NAME, T, SORTED)                      \
  template [[host_name(NAME #T)]] kernel void isin_gather<T, SORTED>( \
      device const T*,                                                \
      device const T*,                                                \
      device bool*,                                                   \
      constant uint&,                                                 \
      constant bool&,                                                 \
      device const uint*,                                             \
      uint)

#define REGISTER_ISIN_OPS(T)                           \
  REGISTER_ISIN_OP("isin_", T, false);                 \
  REGISTER_ISIN_OP("isin_sorted_", T, true);           \
  REGISTER_ISIN_MARK_OP("isin_mark_", T, false);       \
  REGISTER_ISIN_MARK_OP("isin_mark_sorted_", T, true); \
  REGISTER_ISIN_GATHER_OP("isin_gather_", T, false);   \
  REGISTER_ISIN_GATHER_OP("isin_gather_sorted_", T, true)

REGISTER_ISIN_OPS(float);
REGISTER_ISIN_OPS(half);
REGISTER_ISIN_OPS(bfloat);
REGISTER_ISIN_OPS(long);
REGISTER_ISIN_OPS(int);
REGISTER_ISIN_OPS(short);
REGISTER_ISIN_OPS(char);
REGISTER_ISIN_OPS(uchar);
REGISTER_ISIN_OPS(ushort);
REGISTER_ISIN_OPS(uint);
REGISTER_ISIN_OPS(ulong);
