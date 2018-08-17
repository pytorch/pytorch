#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

/// Contains the implementation of parallel reductions in TensorIterator.

namespace at {

using loop2d_t = TensorIterator::loop2d_t;

static bool use_two_pass_reduction(TensorIterator& iter);
static void two_pass_reduction(TensorIterator& iter, const loop2d_t& loop);
static void parallel_dim_reduction(TensorIterator& iter, const loop2d_t& loop);

void TensorIterator::parallel_reduce(const loop2d_t& loop) {
  AT_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
  int64_t numel = this->numel();
  if (numel < at::internal::GRAIN_SIZE) {
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    two_pass_reduction(*this, loop);
  } else {
    parallel_dim_reduction(*this, loop);
  }
}

static bool use_two_pass_reduction(TensorIterator& iter) {
  return iter.tensor(0).numel() == 1;
}

static void two_pass_reduction(TensorIterator& iter, const loop2d_t& loop) {
  int num_threads = at::get_max_threads();

  auto& dst = iter.tensor(0);
  auto buffer_shape = DimVector(dst.sizes());
  buffer_shape.insert(buffer_shape.begin(), num_threads);
  auto buffer = dst.type().tensor(buffer_shape);

  at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    auto slice = buffer[at::get_thread_num()];
    slice.copy_(dst);

    auto sub_iter = TensorIterator::reduce_op(slice, iter.tensor(1));
    sub_iter->serial_for_each(loop, {begin, end});
  });

  auto unsqueezed = dst.unsqueeze(0);
  auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
  final_reduce->for_each(loop);
}

/// Chooses a dimension over which to parallelize. Prefers the outer-most
/// dimension thats larger than the number of available threads.
static int find_split_dim(TensorIterator& iter) {
  int num_threads = at::get_max_threads();
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
round_columns(TensorIterator& iter, int dim, int64_t begin, int64_t end) {
  int64_t elems_per_128_bytes = 128 / iter.element_size(dim);
  begin = begin - (begin % elems_per_128_bytes);
  if (end != iter.shape()[dim]) {
    // only round the 'end' column down if it's not the final column
    end = end - (end % elems_per_128_bytes);
  }
  return std::make_tuple(begin, end);
}

static void parallel_dim_reduction(TensorIterator& iter, const loop2d_t& loop) {
  AT_ASSERT(iter.ndim() >= 1);
  int dim = find_split_dim(iter);
  int64_t cols = iter.shape()[dim];

  bool should_round_columns = iter.strides(1)[dim] == iter.element_size(1);
  at::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
    if (should_round_columns) {
      std::tie(begin, end) = round_columns(iter, dim, begin, end);
    }
    auto sub_iter = TensorIterator(iter);
    sub_iter.narrow(dim, begin, end - begin);
    sub_iter.for_each(loop);
  });
}

}  // namespace at
