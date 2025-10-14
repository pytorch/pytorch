#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at {
namespace detail {
// We prefer to convert through float for reduced-precision floating
// point types if we have a Vectorized specialization for float and we
// don't have one for the actual type in question.
template <typename T>
struct should_prefer_converting_through_float
    : std::bool_constant<
          is_reduced_floating_point_v<T> &&
          vec::is_vec_specialized_for_v<float> &&
          !vec::is_vec_specialized_for_v<T>> {};

template <typename T>
constexpr auto should_prefer_converting_through_float_v =
    should_prefer_converting_through_float<T>::value;
} // namespace detail

namespace vec {
// slow path
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    vec::Vectorized<scalar_t> acc_vec,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  scalar_t acc_arr[Vec::size()];
  acc_vec.store(acc_arr);
  for (const auto i : c10::irange(1, size)) {
    std::array<scalar_t, Vec::size()> acc_arr_next = {0};
    acc_arr_next[0] = acc_arr[i];
    Vec acc_vec_next = Vec::loadu(acc_arr_next.data());
    acc_vec = vec_fun(acc_vec, acc_vec_next);
  }
  acc_vec.store(acc_arr);
  return acc_arr[0];
}

template <typename scalar_t, typename Op>
struct VecReduceAllSIMD {
  static inline scalar_t apply(
      const Op& vec_fun,
      const Vectorized<scalar_t>& acc_vec) {
    return vec_reduce_all(vec_fun, acc_vec, Vectorized<scalar_t>::size());
  }
};

#if defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) && \
    !defined(C10_MOBILE)
#if defined(CPU_CAPABILITY_AVX2)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(
      const Op& vec_fun,
      const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;
    // 128-bit shuffle
    Vec v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm256_cvtss_f32(v);
  }
};
#endif // defined(CPU_CAPABILITY_AVX2)
#if defined(CPU_CAPABILITY_AVX512)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(
      const Op& vec_fun,
      const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;
    // 256-bit shuffle
    Vec v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm512_cvtss_f32(v);
  }
};
#endif // defined(CPU_CAPABILITY_AVX512)
#endif // defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) &&
       // !defined(C10_MOBILE)

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && \
    !defined(CPU_CAPABILITY_SVE)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(
      const Op& vec_fun,
      const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;

    // 64-bit shuffle: [a1+a5, a2+a6, a3+a7, a4+a8, -, -, -, -] -> [a3+a7,
    // a4+a8, a1+a5, a2+a6, -, -, -, -]
    float32x4_t v1_1 = vextq_f32(v, v, 2);
    Vec v1 = v1_1;
    // [a1+a3+a5+a7, a2+a4+a6+a8, a1+a3+a5+a7, a2+a4+a6+a8, -, -, -, -]
    v = vec_fun(v, v1);

    // 32-bit shuffle: [a1+a3+a5+a7, a2+a4+a6+a8, a1+a3+a5+a7, a2+a4+a6+a8, -,
    // -, -, -] -> [a2+a4+a6+a8, a1+a3+a5+a7, a2+a4+a6+a8, a1+a3+a5+a7, -, -, -,
    // -]
    v1_1 = vrev64q_f32(v);
    v1 = v1_1;
    // [a1+a2+a3+a4+a5+a6+a7+a8, a1+a2+a3+a4+a5+a6+a7+a8,
    // a1+a2+a3+a4+a5+a6+a7+a8, a1+a2+a3+a4+a5+a6+a7+a8, -, -, -, -]
    v = vec_fun(v, v1);

    return v[0];
  }
};

template <>
struct VecReduceAllSIMD<float, std::plus<Vectorized<float>>> {
  static inline float apply(
      const std::plus<Vectorized<float>>& vec_fun,
      const Vectorized<float>& acc_vec) {
    return vaddvq_f32(acc_vec);
  }
};
#endif // defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
       // && !defined(CPU_CAPABILITY_SVE)

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && \
    defined(CPU_CAPABILITY_SVE256)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(
      const Op& vec_fun,
      const Vectorized<float>& acc_vec) {
    using Vec = Vectorized<float>;
    Vec v = acc_vec;
    // 128-bit shuffle
    svuint32_t ind = svdupq_n_u32(4, 5, 6, 7);
    Vec v1 = svtbl_f32(v, ind);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    ind = svdupq_n_u32(2, 3, 0, 1);
    v1 = svtbl_f32(v, ind);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    ind = svdupq_n_u32(1, 0, 2, 3);
    v1 = svtbl_f32(v, ind);
    v = vec_fun(v, v1);
    return svlasta(svpfalse(), v);
  }
};
#endif // defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
       // && defined(CPU_CAPABILITY_SVE256)

template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    const Vectorized<scalar_t>& acc_vec) {
  return VecReduceAllSIMD<scalar_t, Op>::apply(vec_fun, acc_vec);
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t reduce_all(
    const Op& vec_fun,
    const scalar_t* data,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
  int64_t d = Vec::size();
  Vec acc_vec = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec = vec_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(vec_fun, acc_vec);
}

// similar to reduce_all, but reduces into two outputs
template <
    typename scalar_t,
    typename Op1,
    typename Op2,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline std::pair<scalar_t, scalar_t> reduce2_all(
    const Op1& vec_fun1,
    const Op2& vec_fun2,
    const scalar_t* data,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    auto loaded_data = Vec::loadu(data, size);
    return std::pair<scalar_t, scalar_t>(
        vec_reduce_all(vec_fun1, loaded_data, size),
        vec_reduce_all(vec_fun2, loaded_data, size));
  }
  int64_t d = Vec::size();
  Vec acc_vec1 = Vec::loadu(data);
  Vec acc_vec2 = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec1 = vec_fun1(acc_vec1, data_vec);
    acc_vec2 = vec_fun2(acc_vec2, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec1 = Vec::set(acc_vec1, vec_fun1(acc_vec1, data_vec), size - d);
    acc_vec2 = Vec::set(acc_vec2, vec_fun2(acc_vec2, data_vec), size - d);
  }
  return std::pair<scalar_t, scalar_t>(
      vec_reduce_all(vec_fun1, acc_vec1), vec_reduce_all(vec_fun2, acc_vec2));
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(red_fun, map_fun(Vec::loadu(data, size)), size);
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    data_vec = map_fun(data_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    data_vec = map_fun(data_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    data_vec = map_fun(data_vec, data2_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline scalar_t map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    const scalar_t* data3,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    Vec data3_vec = Vec::loadu(data3, size);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }

  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    Vec data3_vec = Vec::loadu(data3 + d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    Vec data3_vec = Vec::loadu(data3 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !detail::should_prefer_converting_through_float_v<scalar_t> &&
            std::is_invocable_v<Op, vec::Vectorized<scalar_t>>,
        int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d));
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
    output_vec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !detail::should_prefer_converting_through_float_v<scalar_t> &&
            std::is_invocable_v<
                Op,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>>,
        int> = 0>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(input_data + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(input_data + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !detail::should_prefer_converting_through_float_v<scalar_t> &&
            std::is_invocable_v<
                Op,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>>,
        int> = 0>
inline void map3(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !detail::should_prefer_converting_through_float_v<scalar_t> &&
            std::is_invocable_v<
                Op,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>,
                vec::Vectorized<scalar_t>>,
        int> = 0>
inline void map4(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    const scalar_t* input_data4,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec data_vec4 = Vec::loadu(input_data4 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec data_vec4 = Vec::loadu(input_data4 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d, size - d);
  }
}

} // namespace vec
} // namespace at
