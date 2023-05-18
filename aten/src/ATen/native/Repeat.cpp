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

namespace at {
namespace native {

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

Tensor repeat_interleave(
    const Tensor& self,
    const Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
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
        "repeats must have the same size as input along dim")
  } else {
    AT_ERROR("repeats must be 0-dim or 1-dim tensor");
  }

  auto ret = input.index_select(
      dim.value(), at::repeat_interleave(repeats_, output_size));
  // Restore conj and neg bits
  if (conj) {
    ret = ret.conj();
  }
  if (neg) {
    ret = ret._neg_view();
  }
  return ret;
}

static Tensor repeat_interleave(
    const Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  Tensor input = self;
  at::Tensor repeats_ = at::empty(1, self.options().dtype(at::kLong)).fill_(repeats);
  if (!output_size) {
    if (!dim) {
      input = input.flatten();
      dim = 0;
    }
    auto input_size = input.sym_size(dim.value()).guard_int(__FILE__, __LINE__);
    output_size = input_size * repeats;
  }
  return at::native::repeat_interleave(input, repeats_, dim, output_size);
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
    return at::native::repeat_interleave(self, repeats.guard_int(__FILE__, __LINE__), dim, output_size);
  }

} // namespace native
} // namespace at
