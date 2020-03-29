// Returns the frequency of elements of input non-negative integer tensor.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <tuple>

namespace at { namespace native {

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
  int64_t nbins = static_cast<int64_t>(*self.max().data_ptr<input_t>()) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  const input_t* self_p = self.data_ptr<input_t>();
  if (has_weights) {
    output = native::zeros({nbins}, weights.options());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.data_ptr<weights_t>();
    for (int64_t i = 0; i < self.size(0); i++) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = native::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    for (int64_t i = 0; i < self.size(0); i++) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

Tensor
_bincount_cpu(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self.contiguous(), weights.contiguous(), minlength);
    return _bincount_cpu_template<scalar_t, double>(
        self.contiguous(), weights.contiguous().to(kDouble), minlength);
  });
}

Tensor& histc_out(Tensor& hist, const Tensor &self, int64_t nbins, Scalar minvalue, Scalar maxvalue) {
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  hist.resize_({nbins});
  hist.zero_();

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histc_cpu", [&]() -> void {
    scalar_t minval;
    scalar_t maxval;
    scalar_t *h_data;
    scalar_t *self_data;

    minval = minvalue.to<scalar_t>();
    maxval = maxvalue.to<scalar_t>();

    if (minval == maxval) {
      minval = at::min(self).item().to<scalar_t>();
      maxval = at::max(self).item().to<scalar_t>();
    }
    if (minval == maxval) {
      minval = minval - 1;
      maxval = maxval + 1;
    }

    TORCH_CHECK(!(std::isinf(minval) || std::isinf(maxval) || std::isnan(minval) || std::isnan(maxval)), "range of [", minval, ", ", maxval, "] is not finite");
    TORCH_CHECK(minval < maxval, "max must be larger than min");

    h_data = hist.data_ptr<scalar_t>();
    self_data = self.data_ptr<scalar_t>();

    for(int64_t i=0; i < self.numel(); i++) {
      if (self_data[i] >= minval && self_data[i] <= maxval) {
        const int64_t bin = (int64_t)((self_data[i]-minval) / (maxval-minval) * nbins);
        h_data[std::min(bin, nbins-1)] += 1;
      }
    }
  });

  return hist;
}

Tensor histc(const Tensor& self, int64_t nbins, Scalar minvalue, Scalar maxvalue) {
  Tensor hist = at::empty({0}, self.options());
  at::histc_out(hist, self, nbins, minvalue, maxvalue);
  return hist;
}

}} // namespace at::native
