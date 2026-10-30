#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/SumKernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/functional.h>
#include <algorithm>
#include <array>
#include <functional>

namespace at::native {
namespace {

// Load vector from a smaller type (more elements) to a larger type (fewer elements),
// reducing neighboring elements until it fits into the vector size.
template <typename acc_t, typename scalar_t, typename F>
Vectorized<acc_t> load_reduce_vec(const scalar_t* data, F reduce, acc_t ident) {
  using vec_t = Vectorized<scalar_t>;
  using vacc_t = Vectorized<acc_t>;
  static_assert(vacc_t::size() <= vec_t::size());
  const auto val = vec_t::loadu(data);
  alignas(64) std::array<scalar_t, vec_t::size()> values;
  val.store(values.data());

  constexpr int vstride = vec_t::size() / vacc_t::size();
  alignas(64) std::array<acc_t, vacc_t::size()> acc;
  acc.fill(ident);
  for (const auto k : c10::irange(vstride)) {
    for (const auto i : c10::irange(vacc_t::size())) {
      acc[i] = reduce(acc[i], values[i * vstride + k]);
    }
  }

  return vacc_t::loadu(acc.data());
}

template <typename scalar_t>
struct LoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t*>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadPolicy<Vectorized<scalar_t>> {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * Vectorized<scalar_t>::size();
  }

  static Vectorized<scalar_t> load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = data + index * stride;
    return Vectorized<scalar_t>::loadu(ptr);
  }
};

/* When summing float16 or BFloat16, addition has to be performed in float since
 * that's all the hardware supports. These cast-load policies ensure the entire sum
 * loop is done in float which improves both performance and accuracy.
 */

template <typename scalar_t, typename acc_t>
struct CastLoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    const auto val = LoadPolicy<scalar_t>::load(data, stride, index);
    return acc_t(val);
  }
};

template <typename scalar_t>
struct CastLoadPolicy<scalar_t, scalar_t>:
    LoadPolicy<scalar_t> {
};

// For inner sum, load full vec_t then sum partials down to vacc_t size
template <typename vec_t, typename vacc_t, typename = void>
struct InnerSumCastLoadPolicy;

template <typename vec_t, typename vacc_t>
struct InnerSumCastLoadPolicy <vec_t, vacc_t,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {
  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    return load_reduce_vec<acc_t>(ptr, [](acc_t a, scalar_t b) {
      return a + b;
    }, acc_t(0));
  }
};

template <typename scalar_t>
struct InnerSumCastLoadPolicy<scalar_t, scalar_t, void>:
    LoadPolicy<scalar_t> {
};

template <typename vec_t, typename vacc_t>
struct InnerSumCastLoadPolicy <vec_t, vacc_t, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  using scalar_t = vechold_type<vec_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    vacc_t first, second;
    vec::load_to_float<scalar_t>(ptr, first, second);
    return first + second;
  }
};

// For outer sum, load a partial vec_t of size vacc_t then cast to vacc_t
template <typename vec_t, typename vacc_t, typename = void>
struct OuterSumCastLoadPolicy;

template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy <vec_t, vacc_t,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {

  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * vacc_t::size();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    static_assert(vacc_t::size() <= vec_t::size());
    const auto val = vec_t::loadu(data + stride * index, vacc_t::size());
    alignas(64) scalar_t values[vec_t::size()];
    val.store(values);

    alignas(64) acc_t acc[vacc_t::size()];
    for (const auto i : c10::irange(vacc_t::size())) {
      acc[i] = values[i];
    }

    return vacc_t::loadu(acc);
  }
};

template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy <vec_t, vacc_t, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  using scalar_t = vechold_type<vec_t>;

  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * vacc_t::size();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    vacc_t values;
    vec::load_to_float<scalar_t>(ptr, values);
    return values;
  }
};

template <typename scalar_t>
struct OuterSumCastLoadPolicy<scalar_t, scalar_t, void>:
    LoadPolicy<scalar_t> {
};

/* nansum and the vector norms sum a function of each element: nansum masks out
 * nans, the norms take |x|, x * x or (x != 0). Map is applied to each loaded
 * element, in the accumulation type, before it enters the normal sum loop.
 */

struct NanToZeroMap {
  template <typename T>
  static T apply(T x) {
    return at::_isnan(x) ? T(0) : x;
  }
  template <typename T>
  static Vectorized<T> apply(Vectorized<T> x) {
    return Vectorized<T>::blendv(x, Vectorized<T>(0), x.isnan());
  }
};

struct AbsMap {
  template <typename T>
  static T apply(T x) {
    return std::abs(x);
  }
  template <typename T>
  static Vectorized<T> apply(Vectorized<T> x) {
    return x.abs();
  }
};

struct SquareMap {
  template <typename T>
  static T apply(T x) {
    return x * x;
  }
  template <typename T>
  static Vectorized<T> apply(Vectorized<T> x) {
    return x * x;
  }
};

struct NonZeroMap {
  template <typename T>
  static T apply(T x) {
    return x != T(0) ? T(1) : T(0);
  }
  template <typename T>
  static Vectorized<T> apply(Vectorized<T> x) {
    return x.ne(Vectorized<T>(0));
  }
};

template <typename scalar_t, typename Map>
struct MapLoadPolicy {
  static constexpr int64_t memsize() {
    return LoadPolicy<scalar_t>::memsize();
  }

  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    return Map::apply(LoadPolicy<scalar_t>::load(data, stride, index));
  }
};

template <typename scalar_t, typename acc_t, typename Map>
struct MapCastLoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    return Map::apply(CastLoadPolicy<scalar_t, acc_t>::load(data, stride, index));
  }
};

template <typename vec_t, typename vacc_t, typename Map, typename = void>
struct InnerMapCastLoadPolicy;

template <typename vec_t, typename vacc_t, typename Map>
struct InnerMapCastLoadPolicy <vec_t, vacc_t, Map,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {
  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    return load_reduce_vec<acc_t>(ptr, [](acc_t a, scalar_t b) {
      return a + Map::apply(acc_t(b));
    }, acc_t(0));
  }
};

template <typename scalar_t, typename Map>
struct InnerMapCastLoadPolicy<scalar_t, scalar_t, Map, void>:
    MapLoadPolicy<scalar_t, Map> {
};

template <typename vec_t, typename vacc_t, typename Map>
struct InnerMapCastLoadPolicy <vec_t, vacc_t, Map, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  using scalar_t = vechold_type<vec_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    vacc_t first, second;
    vec::load_to_float<scalar_t>(ptr, first, second);
    return Map::apply(first) + Map::apply(second);
  }
};

template <typename vec_t, typename vacc_t, typename Map>
struct OuterMapCastLoadPolicy {
  static constexpr int64_t memsize() {
    return OuterSumCastLoadPolicy<vec_t, vacc_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    return Map::apply(OuterSumCastLoadPolicy<vec_t, vacc_t>::load(data, stride, index));
  }
};

template <typename scalar_t, typename acc_t>
struct CastStoreAccumulate {
  static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, acc_t value) {
    auto * ptr = reinterpret_cast<scalar_t*>(data + index * stride);
    *ptr += value;
  }
};

template <typename StorePolicy, typename scalar_t>
void store(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  StorePolicy::store(data, stride, index, value);
}

template <typename StorePolicy, typename scalar_t, size_t numel>
void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  for (const auto k : c10::irange(numel)) {
    auto val = values[k];
    StorePolicy::store(base_ptr, stride, k, val);
  }
}

template <typename StorePolicy, typename scalar_t>
void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const Vectorized<scalar_t> &values) {
  using vec_t = Vectorized<scalar_t>;
  alignas(64) std::array<scalar_t, vec_t::size()> array_values{};
  values.store(array_values.data());
  store<StorePolicy>(data, stride, index, array_values);
}

/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single sum
value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  scalar_t row_sum(const scalar_t * data, int64_t n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr int64_t min_chunk_size = 16;

    scalar_t sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (const auto i : c10::irange(n)) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const int64_t chunk_size = std::max(divup(n, min_chunk_size), min_chunk_size);
    for (int64_t i = 0; i < n; i += chunk_size) {
      sum += row_sum(data + i, std::min(chunk_size, n - i));
    }
    return sum;
  }
*/
template <typename scalar_t, int64_t nrows, typename LoadPolicy>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  std::array<std::array<scalar_t, nrows>, num_levels> acc{};
  for (auto &row:acc) {
    row.fill(scalar_t(0));
  }

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char * sum_base = in_data + i * row_stride;
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (const auto k : c10::irange(nrows)) {
        acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
      }
    }

    for (const auto j : c10::irange(1, num_levels)) {
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (const auto k : c10::irange(nrows)) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char * sum_base = in_data + i * row_stride;
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto k : c10::irange(nrows)) {
      acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
    }
  }

  for (const auto j : c10::irange(1, num_levels)) {
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto k : c10::irange(nrows)) {
      acc[0][k] += acc[j][k];
    }
  }
  return acc[0];
}

template <typename scalar_t, typename LoadPolicy>
scalar_t row_sum(const char * C10_RESTRICT in_data,
                 const int64_t in_stride, const int64_t size) {
  constexpr int64_t ilp_factor = 4;

  // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor, LoadPolicy>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += LoadPolicy::load(in_data, in_stride, i);
  }

  for (const auto k : c10::irange(1, ilp_factor)) {
    partial_sums[0] += partial_sums[k];
  }

  return partial_sums[0];
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy, typename StorePolicy>
void vectorized_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t outer_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vacc_t = Vectorized<acc_t>;
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  constexpr int64_t vec_numel = vec_stride / scalar_stride;
  const int64_t vec_size = size0 / vec_numel;

#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  constexpr int64_t ilp_factor = 8;
  constexpr int64_t tail_ilp_factor = 2;
#else
  constexpr int64_t ilp_factor = 4;
  constexpr int64_t tail_ilp_factor = 1;
#endif

  const int64_t size_ilp = vec_size / ilp_factor;

  // Input is contiguous over the first (reduced) dimension
  for (const auto j : c10::irange(size1)) {
    const auto *row_in = data[1] + j * outer_stride;

    // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
    auto partial_sums = multi_row_sum<vacc_t, ilp_factor, VecLoadPolicy>(
        row_in, vec_stride * ilp_factor, vec_stride, size_ilp);

    int64_t i = size_ilp * ilp_factor;
    for (; (i + tail_ilp_factor) <= vec_size; i += tail_ilp_factor) {
      for (int64_t h = 0; h < tail_ilp_factor; h += 1) {
        partial_sums[h] += VecLoadPolicy::load(row_in, vec_stride, i + h);
      }
    }

    for (int64_t k = tail_ilp_factor; k < ilp_factor; k += tail_ilp_factor) {
      for (int64_t h = 0; h < tail_ilp_factor; h += 1) {
        partial_sums[h] += partial_sums[k + h];
      }
    }

    acc_t final_acc = 0;
    for (int64_t k = i * vec_numel; k < size0; ++k) {
      final_acc += ScalarLoadPolicy::load(row_in, scalar_stride, k);
    }

    alignas(64) std::array<acc_t, vacc_t::size() * tail_ilp_factor> partials{};
    for (int64_t h = 0; h < tail_ilp_factor; h += 1) {
      partial_sums[h].store(partials.data() + vacc_t::size() * h);
    }
    for (const auto& partial : partials) {
      final_acc += partial;
    }
    store<StorePolicy>(data[0], out_stride, j, final_acc);
  }
}

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  for (const auto j : c10::irange(size1)) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy, typename StorePolicy>
void vectorized_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vacc_t = Vectorized<acc_t>;
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();

#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  constexpr int64_t nrows = 8;
#else
  constexpr int64_t nrows = 4;
#endif

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vacc_t::size() <= size1; j += nrows * vacc_t::size()) {
    const auto *row_in = data[1] + j * scalar_stride;
    auto sums = multi_row_sum<vacc_t, nrows, VecLoadPolicy>(
        row_in, inner_stride, vec_stride, size0);

    for (const auto i : c10::irange(nrows)) {
      const int64_t base_idx = j + i * vacc_t::size();
      store<StorePolicy>(data[0], out_stride, base_idx, sums[i]);
    }
  }

  for (; j + vacc_t::size() <= size1; j += vacc_t::size()) {
    const auto *row_in = data[1] + j * scalar_stride;
    const vacc_t sums = row_sum<vacc_t, VecLoadPolicy>(
        row_in, inner_stride, size0);

    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * scalar_stride;
    auto ans = row_sum<acc_t, ScalarLoadPolicy>(row_in, inner_stride, size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  constexpr int64_t nrows = 4;
  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<acc_t, nrows, LoadPolicy>(
        row_in, in_strides[0], in_strides[1], size0);
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(
        row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

// Plain sum: elements enter the sum loop unchanged.
struct IdentityMap {};

// Custom floating point sum for better accuracy
template <typename Map, typename scalar_t, typename out_t = scalar_t>
void cascade_sum(TensorIterator &iter, bool sqrt_result = false) {
  constexpr bool is_identity = std::is_same_v<Map, IdentityMap>;
  static_assert(!is_identity || std::is_same_v<scalar_t, out_t>);
  auto loop =
    [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      int64_t in_strides[] = { strides[1], strides[3] };
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      int64_t out_strides[] = { strides[0], strides[2] };

      // Move reduction to be the 1st dim
      if (out_strides[0] != 0 && out_strides[1] == 0) {
        std::swap(in_strides[0], in_strides[1]);
        std::swap(out_strides[0], out_strides[1]);
        std::swap(size0, size1);
      }

      // Special case? - not a true reduction
      if (out_strides[0] != 0 && out_strides[1] != 0) {
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
        int64_t outer_strides[] = { strides[2], strides[3] };
        UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          char* ptrs[3] = { data[0], data[0], data[1] };
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
          if constexpr (is_identity) {
              basic_loop(ptrs, inner_strides, 0, size0,
                         [](scalar_t a, scalar_t b) { return a + b; });
          } else {
              using acc_t = at::acc_type<scalar_t, true>;
              basic_loop(ptrs, inner_strides, 0, size0, [](out_t a, scalar_t b) {
                return out_t(acc_t(a) + Map::apply(acc_t(b)));
              });
          }
        });
        return;
      }

      const int64_t out_stride = out_strides[1];
      TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

      using vec_t = Vectorized<scalar_t>;
      using acc_t = at::acc_type<scalar_t, true>;
      using vacc_t = Vectorized<acc_t>;
      using ScalarLoadPolicy = std::conditional_t<
          is_identity,
          CastLoadPolicy<scalar_t, acc_t>,
          MapCastLoadPolicy<scalar_t, acc_t, Map>>;
      using StorePolicy = CastStoreAccumulate<out_t, acc_t>;

      if (in_strides[0] == sizeof(scalar_t) && size0 >= vec_t::size()) {
        // Contiguous inner reduction
        using VecLoadPolicy = std::conditional_t<
            is_identity,
            InnerSumCastLoadPolicy<vec_t, vacc_t>,
            InnerMapCastLoadPolicy<vec_t, vacc_t, Map>>;
        vectorized_inner_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
            data, in_strides[1], out_stride, size0, size1);
      } else if (in_strides[1] == sizeof(scalar_t) && size1 >= vec_t::size()) {
        // Contiguous outer reduction
        using VecLoadPolicy = std::conditional_t<
            is_identity,
            OuterSumCastLoadPolicy<vec_t, vacc_t>,
            OuterMapCastLoadPolicy<vec_t, vacc_t, Map>>;
        vectorized_outer_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
            data, in_strides[0], out_stride, size0, size1);
      } else if (in_strides[0] < in_strides[1]) {
        scalar_inner_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
            data, in_strides, out_stride, size0, size1);
      } else {
        scalar_outer_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
            data, in_strides, out_stride, size0, size1);
      }
    };

  if constexpr (!is_identity) {
    // TensorIterator::parallel_reduce would add up the per-thread results of a
    // reduction to one element with this same loop, applying Map to them again
    // (and reading them as scalar_t). The output is written once here, so it
    // can also be a reduced precision scalar_t.
    if (iter.output_base(0).numel() == 1) {
      using acc_t = at::acc_type<scalar_t, true>;
      const acc_t total = at::parallel_reduce(
          0, iter.numel(), at::internal::GRAIN_SIZE, acc_t(0),
          [&](int64_t begin, int64_t end, acc_t /*ident*/) {
            out_t partial(0);
            iter.serial_for_each(
                [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
                  char* ptrs[] = {reinterpret_cast<char*>(&partial), data[1]};
                  loop(ptrs, strides, size0, size1);
                },
                {begin, end});
            return acc_t(partial);
          },
          std::plus<acc_t>());
      const acc_t result = sqrt_result ? std::sqrt(total) : total;
      if (iter.dtype(0) == c10::CppTypeToScalarType<out_t>::value) {
        *static_cast<out_t*>(iter.data_ptr(0)) = out_t(result);
      } else {
        *static_cast<scalar_t*>(iter.data_ptr(0)) = scalar_t(result);
      }
      return;
    }
  }
  iter.output_base().fill_(out_t(0));
  iter.parallel_reduce(loop);
  if (sqrt_result) {
    const TensorBase& out = iter.output_base();
    auto sqrt_iter = TensorIteratorConfig().add_output(out).add_const_input(out).build();
    cpu_kernel(sqrt_iter, [](out_t a) -> out_t { return std::sqrt(a); });
  }
}

void sum_kernel_impl(TensorIterator &iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
      [&] {
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a + b; });
      });
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu", [&] {
    cascade_sum<IdentityMap, scalar_t>(iter);
  });
}

void nansum_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "nansum_cpu", [&] {
    cascade_sum<NanToZeroMap, scalar_t>(iter);
  });
}

template <typename scalar_t, typename out_t>
void cascade_powsum(TensorIterator &iter, double p, bool is_norm) {
  if (p == 0) {
    cascade_sum<NonZeroMap, scalar_t, out_t>(iter);
  } else if (p == 1) {
    cascade_sum<AbsMap, scalar_t, out_t>(iter);
  } else {
    TORCH_INTERNAL_ASSERT(p == 2, "cascade_powsum: unsupported p ", p);
    cascade_sum<SquareMap, scalar_t, out_t>(iter, /*sqrt_result=*/is_norm);
  }
}

}  // namespace (anonymous)

inline namespace CPU_CAPABILITY {

void cascade_powsum_kernel(TensorIterator &iter, double p, bool is_norm) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.input_dtype(), "cascade_powsum_cpu", [&] {
    using out_t = std::conditional_t<is_reduced_floating_point_v<scalar_t>, float, scalar_t>;
    TORCH_INTERNAL_ASSERT(iter.dtype(0) == c10::CppTypeToScalarType<out_t>::value ||
                          (iter.dtype(0) == iter.input_dtype() && iter.output_base(0).numel() == 1),
                          "cascade_powsum: unexpected output dtype ", iter.dtype(0));
    cascade_powsum<scalar_t, out_t>(iter, p, is_norm);
  });
}

}  // namespace CPU_CAPABILITY

// nansum on Float16 has poor accuracy with AVX2, and more so with AVX512.
// So until it's fixed, it won't be dispatched with AVX512. GH issue 59415.
// Besides, these kernels are slower with AVX512 than with AVX2.
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_impl)
REGISTER_DISPATCH(sum_stub, &sum_kernel_impl)

}  // namespace at::native
