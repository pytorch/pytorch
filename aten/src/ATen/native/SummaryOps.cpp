// Returns the frequency of elements of input non-negative integer tensor.
#include <ATen/native/SummaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <tuple>

namespace at {
namespace meta {

TORCH_META_FUNC(histc)(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max) {
  TORCH_CHECK(!min.isComplex(),
              "Expected min to be real but found complex value ", min.toComplexDouble());
  TORCH_CHECK(!max.isComplex(),
              "Expected max to be real but found complex value ", max.toComplexDouble());
  auto min_v = min.toDouble();
  auto max_v = max.toDouble();
  TORCH_CHECK(min_v <= max_v, "Expected min <= max but found: ", min_v, " > ", max_v);
  TORCH_CHECK(bins > 0, "Expected bins to be a positive integer, but found ", bins);
  set_output(0, IntArrayRef(bins), {}, self.options(), {});
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(histc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

TORCH_IMPL_FUNC(histc_out_cpu)(const Tensor& self, int64_t bins, const Scalar& min,
                               const Scalar& max, const Tensor& result) {
  Scalar minval(min), maxval(max);
  if (minval.equal(maxval)) {
    minval = self.min().item();
    maxval = self.max().item();

    if (minval.equal(maxval)) {
      if (minval.isIntegral(/*includeBool=*/true)) {
        minval = minval.to<int64_t>() - 1;
        maxval = maxval.to<int64_t>() + 1;
      } else {
        minval = minval.to<double>() - 1.0;
        maxval = maxval.to<double>() + 1.0;
      }
    }
  }

  auto iter = TensorIteratorConfig()
    .add_input(self)
    .build();
  result.zero_();
  histc_stub(kCPU, iter, minval, maxval, result);
}


///////////////// bincount /////////////////
namespace {

template <typename input_t, typename weights_t>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return native::zeros({minlength}, kLong);
  }
  if (self.dim() != 1 || *self.min().data_ptr<input_t>() < 0) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }

  Tensor output;
  int64_t self_size = self.size(0);
  int64_t nbins = static_cast<int64_t>(*self.max().data_ptr<input_t>()) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  const input_t* self_p = self.data_ptr<input_t>();
  if (has_weights) {
    output = native::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.data_ptr<weights_t>();
    for (int64_t i = 0; i < self_size; i++) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = native::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    for (int64_t i = 0; i < self_size; i++) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

Tensor
_bincount_cpu(const Tensor& self, const c10::optional<Tensor>& weights_opt, int64_t minlength) {
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

}} // namespace at::native
