// Returns the frequency of elements of input non-negative integer tensor.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>

#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

///////////////// bincount /////////////////
namespace {

template <typename input_t, typename weights_t>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    TORCH_CHECK(false, "minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong);
  }
  if (self.dim() != 1 || *self.min().data_ptr<input_t>() < 0) {
    TORCH_CHECK(false, "bincount only supports 1-d non-negative integral inputs.");
  }

  // Ensure max_val < 2 ^ 63 - 1 (9223372036854775807)
  auto max_val = *self.max().data_ptr<input_t>();
  if (max_val >= std::numeric_limits<int64_t>::max()) {
    TORCH_CHECK(false,
        "maximum value of input overflowed, it should be < ",
        std::numeric_limits<int64_t>::max(),
        " but got ",
        max_val
    );
  }

  bool has_weights = weights.defined();
  if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
    TORCH_CHECK(false, "weights should be 1-d and have the same length as input");
  }

  Tensor output;
  int64_t self_size = self.size(0);
  int64_t nbins = static_cast<int64_t>(max_val) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  const input_t* self_p = self.const_data_ptr<input_t>();
  if (has_weights) {
    output = at::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.const_data_ptr<weights_t>();
    for (const auto i : c10::irange(self_size)) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = at::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    for (const auto i : c10::irange(self_size)) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

Tensor
_bincount_cpu(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self.contiguous(), weights.contiguous(), minlength);
    return _bincount_cpu_template<scalar_t, double>(
        self.contiguous(), weights.contiguous().to(kDouble), minlength);
  });
}

} // namespace at::native
