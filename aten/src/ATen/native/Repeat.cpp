#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Repeat.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/repeat_interleave_native.h>
#endif

template <typename index_t>
static void compute_cpu(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  TORCH_CHECK(
      (result_size == cumsum_ptr[size - 1]),
      "allocated size does not match required size");
  at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
    for (const auto i : c10::irange(i_begin, i_end)) {
      int64_t end = cumsum_ptr[i];
      index_t size = repeat_ptr[i];
      int64_t start = end - size;
      // A negative repeat makes the cumsum non-monotonic, so even a
      // non-negative element can yield start < 0 or end > result_size and write
      // out of bounds. When output_size is given this per-element check is the
      // only guard against negative repeats, so validate the write range before
      // touching result_ptr.
      TORCH_CHECK(
          size >= 0 && start >= 0 && end <= result_size,
          "repeats can not be negative");
      for (const auto j : c10::irange(start, end)) {
        result_ptr[j] = i;
      }
    }
  });
}

namespace at::native {

Tensor repeat_interleave_cpu(
    const Tensor& repeat,
    std::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INTEGRAL_TYPES(
      repeat.scalar_type(), "repeat_interleave_cpu", [&]() {
        output = repeat_interleave_common<scalar_t, compute_cpu<scalar_t>>(
            repeat, output_size);
      });

  return output;
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    const Tensor& repeats,
    std::optional<int64_t> dim,
    std::optional<SymInt> output_size) {
  Tensor input = self;

  // Store conj and neg bits
  const auto conj = input.is_conj();
  if (conj) {
    input = input.conj();
  }
  const auto neg = input.is_neg();
  if (neg) {
    input = input._neg_view();
  }

  if (!dim) {
    input = input.flatten();
    dim = 0;
  }

  Tensor repeats_ = repeats;
  if (repeats.dim() == 0 || (repeats.dim() == 1 && TORCH_GUARD_OR_FALSE(repeats.sym_size(0).sym_eq(1)))) {
    repeats_ = repeats.reshape({1}).expand_symint({input.sym_size(dim.value())});
  } else if (repeats.dim() == 1) {
    TORCH_CHECK(
        repeats.sym_size(0) == input.sym_size(dim.value()),
        "repeats must have the same size as input along dim, but got repeats.size(0) = ",
        repeats.sym_size(0), " and input.size(", dim.value(), ") = ", input.sym_size(dim.value())
    );
  } else {
    TORCH_CHECK(false, "repeats must be 0-dim or 1-dim tensor");
  }

  Tensor repeat_indices =
      at::repeat_interleave_symint(repeats_, std::move(output_size));
  if (repeat_indices.scalar_type() != at::kLong &&
      repeat_indices.scalar_type() != at::kInt) {
    repeat_indices = repeat_indices.to(at::kLong);
  }
  auto ret = input.index_select(dim.value(), repeat_indices);
  // Restore conj and neg bits
  if (conj) {
    ret = ret.conj();
  }
  if (neg) {
    ret = ret._neg_view();
  }
  return ret;
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    c10::SymInt repeats,
    std::optional<int64_t> dim_opt,
    std::optional<SymInt> output_size) {
  Tensor input = dim_opt ? self : self.flatten();
  int64_t dim = c10::maybe_wrap_dim(dim_opt.value_or(0), self.dim());
  TORCH_SYM_CHECK(repeats.sym_ge(0), "Repeats must be non-negative");

  input = input.unsqueeze(dim + 1);
  auto expand_shape = input.sym_sizes().vec();
  expand_shape[dim + 1] = repeats;
  input = input.expand_symint(expand_shape);

  // This argument doesn't really make sense for the scalar overload, but exists
  // for consistency with the tensor overload
  if (output_size) {
    auto calculated_size = repeats * expand_shape[dim];
    TORCH_SYM_CHECK(
        output_size->sym_eq(calculated_size),
        "repeat_interleave: Invalid output_size, expected ",
        calculated_size,
        " but got ",
        *output_size);
  }

  return input.clone(at::MemoryFormat::Contiguous).flatten(dim, dim + 1);
}

} // namespace at::native
