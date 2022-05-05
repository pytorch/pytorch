#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

// Check that indices fall within dimension array size
// Avoid redispatch call to min/max
template <typename index_t>
static inline void check_indexarray_range(
    const index_t* indices,
    int64_t index_size,
    index_t dim_size) {
  for (const auto i : c10::irange(index_size)) {
    index_t idx = indices[i];
    TORCH_CHECK(
        0 <= idx && idx < dim_size,
        "INDICES element is out of DATA bounds, id=",
        idx,
        " axis_dim=",
        dim_size);
  }
}

template <typename scalar_t>
static inline void copy_stub(scalar_t* result, scalar_t* self, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(self + d);
    data_vec.store(result + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; ++d) {
    result[d] = self[d];
  }
}

template <typename scalar_t, typename index_t>
static void index_select_firstdim_impl(
    scalar_t* result_data,
    scalar_t* self_data,
    index_t* index_data,
    int64_t index_size,
    int64_t inner_size) {
  constexpr int64_t grain_size = internal::GRAIN_SIZE / 2;
  if (inner_size > grain_size) {
    constexpr int64_t block_size = 2048;
    int64_t num_blocks = divup(inner_size, block_size);
    at::parallel_for(0, index_size * num_blocks, grain_size / block_size, [&](int64_t begin, int64_t end) {
      for (const auto ii : c10::irange(begin, end)) {
        int64_t j = ii / num_blocks;
        int64_t k = ii % num_blocks;
        int64_t inner_idx_begin = k * block_size;
        int64_t size = std::min(block_size, inner_size - inner_idx_begin);

        index_t offset = index_data[j];
        scalar_t* self_ptr = self_data + offset * inner_size + inner_idx_begin;
        scalar_t* result_ptr = result_data + j * inner_size + inner_idx_begin;
        copy_stub(result_ptr, self_ptr, size);
      }
    });
  } else {
    at::parallel_for(0, index_size, grain_size / inner_size, [&](int64_t begin, int64_t end) {
      for (const auto j : c10::irange(begin, end)) {
        index_t offset = index_data[j];
        #ifdef __GNUC__
        if (j + 1 < index_size) {
          __builtin_prefetch(self_data + index_data[j + 1] * inner_size, 0, 1);
        }
        #endif // __GNUC__
        scalar_t* self_ptr = self_data + offset * inner_size;
        scalar_t* result_ptr = result_data + j * inner_size;
        copy_stub(result_ptr, self_ptr, inner_size);
      }
    });
  }
}

template <typename scalar_t, typename index_t>
static void index_select_non_firstdim_impl(
    scalar_t* result_data,
    scalar_t* self_data,
    index_t* index_data,
    int64_t outer_size,
    int64_t dim_size,
    int64_t index_size,
    int64_t inner_size) {
  constexpr int64_t grain_size = internal::GRAIN_SIZE / 2;
  at::parallel_for(0, outer_size * index_size, grain_size / inner_size,[&](int64_t begin, int64_t end) {
    int64_t i{0}, j{0};
    // first elem in each thread
    data_index_init(begin, i, outer_size, j, index_size);
    for (const auto ii : c10::irange(begin, end)) {
      index_t offset = index_data[j];
      scalar_t* self_ptr = self_data + i * dim_size * inner_size + offset * inner_size;
      scalar_t* result_ptr = result_data + ii * inner_size;
      copy_stub(result_ptr, self_ptr, inner_size);
      // move on to next index in {outer_size, index_size}
      data_index_step(i, outer_size, j, index_size);
    }
  });
}

template <typename scalar_t, typename index_t, int64_t inner_size>
static void index_select_gather_impl(
    scalar_t* result_data,
    scalar_t* self_data,
    index_t* index_data,
    int64_t outer_size,
    int64_t dim_size,
    int64_t index_size) {
  using Vec = vec::Vectorized<scalar_t>;
  using integer_t = vec::int_same_size_t<scalar_t>;
  using iVec = vec::Vectorized<integer_t>;

  constexpr int64_t grain_size = internal::GRAIN_SIZE / 2;
  int64_t len = index_size - (index_size % Vec::size());
  at::parallel_for(0, outer_size, grain_size / (index_size * inner_size), [&](int64_t begin, int64_t end) {
    // create the offset stencil for each row in outer dimenson,
    // shared across {outer_size}
    std::unique_ptr<integer_t []> index_buffer(new integer_t[index_size * inner_size]);
    for (const auto j : c10::irange(index_size)) {
      for (const auto k : c10::irange(inner_size)) {
        index_buffer[j * inner_size + k] = integer_t(index_data[j] * inner_size + k);
      }
    }
    for (const auto i : c10::irange(begin, end)) {
      scalar_t* self_ptr = self_data + i * dim_size * inner_size;
      scalar_t* result_ptr = result_data + i * index_size * inner_size;

      // `gather` data chunk of Vec::size() by inner_size each step
      int64_t j = 0;
      for (; j < len; j += Vec::size()) {
        for (const auto k : c10::irange(inner_size)) {
          iVec offset_ivec = iVec::loadu(index_buffer.get() + j * inner_size + k * Vec::size());
          Vec out_vec = vec::gather<sizeof(scalar_t)>(self_ptr, offset_ivec);
          out_vec.store(result_ptr + j * inner_size + k * Vec::size());
        }
      }
      for (; j < index_size; ++j) {
        for (const auto k : c10::irange(inner_size)) {
          index_t offset = index_buffer[j * inner_size + k];
          result_ptr[j * inner_size + k] = self_ptr[offset];
        }
      }
    }
  });
}

template <typename scalar_t, typename index_t>
void cpu_index_select_dispatch(const Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  auto self_sizes = self.sizes();
  int64_t outer_size = c10::size_to_dim_(dim, self_sizes);
  int64_t dim_size = self_sizes[dim];
  int64_t inner_size = c10::size_from_dim_(dim + 1, self_sizes);
  int64_t index_size = index.numel();

  // normalize self and result shape as:
  //   self: [outer_size, dim_size, inner_size]
  //   result: [outer_size, index_size, inner_size]
  //
  scalar_t* result_data = result.data_ptr<scalar_t>();
  scalar_t* self_data = self.data_ptr<scalar_t>();
  index_t* index_data = index.data_ptr<index_t>();

  check_indexarray_range<index_t>(index_data, index_size, dim_size);

  // Note on index_select implementation choosen:
  //
  // 1. `index_select_gather_impl`: used when inner_size == 1 or 2.
  //   inner_size == 1 indicates a gather across {index_size}, here
  //   vector gather is used which roughly saved ~20% uops v.s. scalar version.
  //   The kernel parallels on {outer_size}.
  //
  // 2. `index_select_firstdim_impl`: used when dim is the first dimension.
  //   The kernel may directly parallel on {index_size} or do blocking on {inner_size}
  //   so as to further extend parallelism. Therefore we may efficiently handle
  //   both thin tall shapes and flat shapes.
  //
  // 3. `index_select_non_firstdim_impl`: the most generic case.
  //   The kernel parallels on {outer_size, index_size} and do vectorized copy
  //   on {inner_size}
  //
  // Lower the default grain size by half since index_select is indirect memory access.
  //
  int64_t max_value = std::numeric_limits<int32_t>::max();
  bool can_use_32bit_indexing = (dim_size * inner_size) < max_value;

  const auto st = result.scalar_type();
  if (st == kFloat && can_use_32bit_indexing && inner_size == 1) {
    index_select_gather_impl<scalar_t, index_t, 1>(
        result_data, self_data, index_data, outer_size, dim_size, index_size);
  } else if (st == kFloat && can_use_32bit_indexing && inner_size == 2) {
    index_select_gather_impl<scalar_t, index_t, 2>(
        result_data, self_data, index_data, outer_size, dim_size, index_size);
  } else if (outer_size == 1) {
    index_select_firstdim_impl<scalar_t, index_t>(
        result_data, self_data, index_data, index_size, inner_size);
  } else {
    index_select_non_firstdim_impl<scalar_t, index_t>(
        result_data, self_data, index_data, outer_size, dim_size, index_size, inner_size);
  }
}

void index_select_contig_kernel(const Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, result.scalar_type(),
      "index_select_contig", [&result, &self, &dim, &index] {
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "cpu_index_select_contig", [&result, &self, &dim, &index] {
      cpu_index_select_dispatch<scalar_t, index_t>(result, self, dim, index);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(index_select_contig_stub, &index_select_contig_kernel);

}} // at::native
