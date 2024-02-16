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
    index_t* repeat_ptr,
    int64_t* cumsum_ptr,
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
      TORCH_CHECK((size >= 0), "repeats can not be negative");
      int64_t start = end - size;
      for (const auto j : c10::irange(start, end)) {
        result_ptr[j] = i;
      }
    }
  });
}

namespace at::native {

Tensor repeat_interleave_cpu(
    const Tensor& repeat,
    c10::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_cpu", [&]() {
    output = repeat_interleave_common<index_t, compute_cpu<index_t>>(
        repeat, output_size);
  });

  return output;
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    const Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<SymInt> output_size) {
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
  if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.sym_size(0) == 1)) {
    repeats_ = repeats.reshape({1}).expand_symint({input.sym_size(dim.value())});
  } else if (repeats.dim() == 1) {
    TORCH_CHECK(
        repeats.sym_size(0) == input.sym_size(dim.value()),
        "repeats must have the same size as input along dim, but got repeats.size(0) = ",
        repeats.sym_size(0), " and input.size(", dim.value(), ") = ", input.sym_size(dim.value())
    );
  } else {
    AT_ERROR("repeats must be 0-dim or 1-dim tensor");
  }

  auto ret = input.index_select(
      dim.value(), at::repeat_interleave_symint(repeats_, output_size));
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
    c10::optional<int64_t> dim_opt,
    c10::optional<SymInt> output_size) {
  Tensor input = dim_opt ? self : self.flatten();
  int64_t dim = c10::maybe_wrap_dim(dim_opt.value_or(0), self.dim());
  TORCH_CHECK(repeats >= 0, "Repeats must be non-negative");

  input = input.unsqueeze(dim + 1);
  auto expand_shape = input.sym_sizes().vec();
  expand_shape[dim + 1] = repeats;
  input = input.expand_symint(expand_shape);

  // This argument doesn't really make sense for the scalar overload, but exists
  // for consistency with the tensor overload
  if (output_size) {
    auto calculated_size = (repeats * expand_shape[dim]).guard_int(__FILE__, __LINE__);
    TORCH_CHECK(*output_size == calculated_size, "repeat_interleave: Invalid output_size, expected ",
                calculated_size, " but got ", *output_size);
  }

  return input.clone(at::MemoryFormat::Contiguous).flatten(dim, dim + 1);
}

} // namespace at::native
