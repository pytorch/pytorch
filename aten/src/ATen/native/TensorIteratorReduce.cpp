#include <ATen/TensorIterator.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <memory>
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorIteratorInternal.h>

#include <c10/util/irange.h>

/// Contains the implementation of parallel reductions in TensorIterator.

namespace at {

using loop2d_t = TensorIteratorBase::loop2d_t;

static bool use_two_pass_reduction(TensorIteratorBase& iter);
static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop);
static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop);

void TensorIteratorBase::parallel_reduce(loop2d_t loop) {
  TORCH_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
  int64_t numel = this->numel();
  if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region()) {
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    two_pass_reduction(*this, loop);
  } else {
    parallel_dim_reduction(*this, loop);
  }
}

static bool use_two_pass_reduction(TensorIteratorBase& iter) {
  return iter.output(0).numel() == 1;
}

static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  const int max_threads = at::get_num_threads();

  auto dst = iter.output(0);
  auto unsqueezed = dst.unsqueeze(0);
  auto buffer_shape = DimVector(unsqueezed.sizes());
  buffer_shape[0] = max_threads;
  auto buffer = at::empty(buffer_shape, dst.options());
  // Fill with the identity
  buffer.copy_(unsqueezed);

  auto buffer_stride = buffer.strides()[0] * buffer.element_size();
  auto buffer_0 = buffer[0];
  auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));
  TORCH_INTERNAL_ASSERT(first_reduce.output(0).is_alias_of(buffer_0));

  at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    const auto thread_num = at::get_thread_num();
    auto shape = first_reduce.shape();
    auto strides = first_reduce.get_strides();

    // Bump output ptr so each thread has its own ouput slice
    auto base_ptrs = first_reduce.get_base_ptrs();
    base_ptrs[0] += buffer_stride * thread_num;

    at::internal::serial_for_each(shape, strides, base_ptrs, loop, {begin, end});
  });

  auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
  final_reduce.for_each(loop);
}

/// Chooses a dimension over which to parallelize. Prefers the outer-most
/// dimension thats larger than the number of available threads.
static int find_split_dim(TensorIteratorBase& iter) {
  int num_threads = at::get_num_threads();
  auto shape = iter.shape();

  // start with the outer-most dimension
  int best_dim = iter.ndim() - 1;
  for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
    if (shape[dim] >= num_threads) {
      return dim;
    } else if (shape[dim] > shape[best_dim]) {
      best_dim = dim;
    }
  }

  AT_ASSERT(!iter.is_dim_reduced(best_dim));
  return best_dim;
}

static std::tuple<int64_t, int64_t>
round_columns(TensorIteratorBase& iter, int dim, int multiple, int64_t begin, int64_t end) {
  begin = begin - (begin % multiple);
  if (end != iter.shape()[dim]) {
    // only round the 'end' column down if it's not the final column
    end = end - (end % multiple);
  }
  return std::make_tuple(begin, end);
}

static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  AT_ASSERT(iter.ndim() >= 1);
  int dim = find_split_dim(iter);
  int64_t cols = iter.shape()[dim];
  int element_size = iter.element_size(/*arg=*/1);

  bool should_round_columns = iter.strides(1)[dim] == element_size;
  at::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
    if (should_round_columns) {
      // round columns to multiples of 128 bytes if adjacent columns are
      // contiguous in memory.
      int64_t cols_per_128_bytes = 128 / element_size;
      std::tie(begin, end) = round_columns(iter, dim, cols_per_128_bytes, begin, end);
    }
    if (begin == end) {
      return;
    }
    auto sub_iter = TensorIterator(iter);
    sub_iter.narrow(dim, begin, end - begin);
    sub_iter.for_each(loop);
  });
}

void TensorIteratorBase::foreach_reduced_elt(loop_subiter_t loop, bool parallelize) {
  AT_ASSERT(ninputs() == 1);
  AT_ASSERT(noutputs() >= 1);

  auto shape = this->shape();
  if (output(0).numel() == 0) {
    return;
  }
  if (output(0).numel() == 1) {
    loop(*this);
  }
  else if (numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region() || !parallelize) {
    auto reduce_dims = num_reduce_dims();

    auto non_reduced_shape = shape.slice(reduce_dims, shape.size() - reduce_dims);

    int64_t non_reduced_numel = 1;
    for (const auto i : c10::irange(non_reduced_shape.size())) {
      non_reduced_numel *= non_reduced_shape[i];
    }
    DimCounter dims {non_reduced_shape, {0, non_reduced_numel}};
    while (!dims.is_done()) {
      TensorIterator reduced = *this;
      reduced.select_all_keeping_dim(reduce_dims, dims.values);
      loop(reduced);
      dims.increment({1, 1});
    }
  }
  else {
    int dim = find_split_dim(*this);
    int64_t cols = shape[dim];
    at::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
      if (begin == end) {
        return;
      }

      TensorIterator sub_iter(*this);

      sub_iter.narrow(dim, begin, end - begin);
      // On some broken setups, `#ifdef _OPENMP` is true,
      // and `get_num_threads` returns > 1, but
      // `#pragma omp parallel` is ignored.
      // There is no API to check for this, so we need to explicitly
      // stop trying to parallelize if we've already gotten here.
      //
      // (If we are on one of those broken setups, we will
      //  only have one thread here, and end - begin == cols.)
      sub_iter.foreach_reduced_elt(loop, false);
    });
  }
}

}  // namespace at
