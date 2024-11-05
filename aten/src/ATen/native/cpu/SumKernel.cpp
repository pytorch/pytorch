#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/functional.h>
#include <algorithm>

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

/* To implement nansum, augment the load operation to mask out nans before
 * entering the normal sum loop.
 */

template <typename scalar_t>
struct NanSumLoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = LoadPolicy<scalar_t>::load(data, stride, index);
    return at::_isnan(val) ? scalar_t(0) : val;
  }
};

template <typename scalar_t>
struct NanSumLoadPolicy<Vectorized<scalar_t>> {
  using vec_t = Vectorized<scalar_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vec_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = LoadPolicy<vec_t>::load(data, stride, index);
    return vec_t::blendv(val, vec_t(0), val.isnan());
  }
};

template <typename scalar_t, typename acc_t>
struct NanSumCastLoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = CastLoadPolicy<scalar_t, acc_t>::load(data, stride, index);
    return at::_isnan(val) ? acc_t(0) : val;
  }
};

template <typename vec_t, typename vacc_t, typename = void>
struct InnerNanSumCastLoadPolicy;

template <typename vec_t, typename vacc_t>
struct InnerNanSumCastLoadPolicy <vec_t, vacc_t,
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
      return at::_isnan(b) ? a : a + b;
    }, acc_t(0));
  }
};

template <typename scalar_t>
struct InnerNanSumCastLoadPolicy<scalar_t, scalar_t, void>:
    NanSumLoadPolicy<scalar_t> {
};

template <typename vec_t, typename vacc_t>
struct InnerNanSumCastLoadPolicy <vec_t, vacc_t, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  using scalar_t = vechold_type<vec_t>;

  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    vacc_t first, second;
    vec::load_to_float<scalar_t>(ptr, first, second);
    const vacc_t zero(0);
    return (vacc_t::blendv(first, zero, first.isnan()) +
            vacc_t::blendv(second, zero, second.isnan()));
  }
};

template <typename vec_t, typename vacc_t>
struct OuterNanSumCastLoadPolicy {
  static constexpr int64_t memsize() {
    return OuterSumCastLoadPolicy<vec_t, vacc_t>::memsize();
  }

  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = OuterSumCastLoadPolicy<vec_t, vacc_t>::load(data, stride, index);
    return vacc_t::blendv(val, vacc_t(0), val.isnan());
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
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  StorePolicy::store(data, stride, index, value);
}

template <typename StorePolicy, typename scalar_t, size_t numel>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  for (const auto k : c10::irange(numel)) {
    auto val = values[k];
    StorePolicy::store(base_ptr, stride, k, val);
  }
}

template <typename StorePolicy, typename scalar_t>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
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

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  scalar_t acc[num_levels][nrows];
  std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));

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

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<scalar_t, nrows> ret;
  for (const auto k : c10::irange(nrows)) {
    ret[k] = acc[0][k];
  }
  return ret;
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

  // Input is contiguous over the first (reduced) dimension
  for (const auto j : c10::irange(size1)) {
    const auto *row_in = data[1] + j * outer_stride;
    auto vec_acc = row_sum<vacc_t, VecLoadPolicy>(row_in, vec_stride, vec_size);

    acc_t final_acc = 0;
    for (int64_t k = vec_size * vec_numel; k < size0; ++k) {
      final_acc += ScalarLoadPolicy::load(row_in, scalar_stride, k);
    }

    alignas(64) std::array<acc_t, vacc_t::size()> partials{};
    vec_acc.store(partials.data());
    for (const auto k : c10::irange(partials.size())) {
      final_acc += partials[k];
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
  constexpr int64_t nrows = 4;

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

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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

// Custom floating point sum for better accuracy
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  iter.output_base().fill_(scalar_t(0));
  iter.parallel_reduce(
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
          if constexpr (ignore_nan) {
              basic_loop(ptrs, inner_strides, 0, size0, [](scalar_t a, scalar_t b) {
                auto a_notnan = at::_isnan(a) ? scalar_t(0) : a;
                auto b_notnan = at::_isnan(b) ? scalar_t(0) : b;
                return a_notnan + b_notnan;
              });
          } else {
              basic_loop(ptrs, inner_strides, 0, size0,
                         [](scalar_t a, scalar_t b) { return a + b; });
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
          ignore_nan,
          NanSumCastLoadPolicy<scalar_t, acc_t>,
          CastLoadPolicy<scalar_t, acc_t>>;
      using StorePolicy = CastStoreAccumulate<scalar_t, acc_t>;

      if (in_strides[0] == sizeof(scalar_t) && size0 >= vec_t::size()) {
        // Contiguous inner reduction
        using VecLoadPolicy = std::conditional_t<
            ignore_nan,
            InnerNanSumCastLoadPolicy<vec_t, vacc_t>,
            InnerSumCastLoadPolicy<vec_t, vacc_t>>;
        vectorized_inner_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
            data, in_strides[1], out_stride, size0, size1);
      } else if (in_strides[1] == sizeof(scalar_t) && size1 >= vec_t::size()) {
        // Contiguous outer reduction
        using VecLoadPolicy = std::conditional_t<
            ignore_nan,
            OuterNanSumCastLoadPolicy<vec_t, vacc_t>,
            OuterSumCastLoadPolicy<vec_t, vacc_t>>;
        vectorized_outer_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
            data, in_strides[0], out_stride, size0, size1);
      } else if (in_strides[0] < in_strides[1]) {
        scalar_inner_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
            data, in_strides, out_stride, size0, size1);
      } else {
        scalar_outer_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
            data, in_strides, out_stride, size0, size1);
      }
    });
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
    cascade_sum</*ignore_nan=*/false, scalar_t>(iter);
  });
}

void nansum_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "nansum_cpu", [&] {
    cascade_sum</*ignore_nan=*/true, scalar_t>(iter);
  });
}

}  // namespace (anonymous)

// nansum on Float16 has poor accuracy with AVX2, and more so with AVX512.
// So until it's fixed, it won't be dispatched with AVX512. GH issue 59415.
// Besides, these kernels are slower with AVX512 than with AVX2.
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_impl)
REGISTER_DISPATCH(sum_stub, &sum_kernel_impl)

}  // namespace at::native
