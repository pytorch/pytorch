#pragma once

#include <ATen/ATen.h>
#include <utility>
#include <ATen/NumericUtils.h>

namespace at {
namespace native {

// This assumes the template being called on a tensor with the correct dtype
template<typename scalar_t>
inline std::pair<scalar_t, scalar_t> _histogram_maybe_compute_range(const Tensor& self,
    c10::optional<ArrayRef<double>> range) {
  scalar_t minvalue;
  scalar_t maxvalue;
  if (range.has_value()) {
    // If range is defined, max must be larger than min.
    TORCH_CHECK(
        range.value()[0] < range.value()[1], "max must be larger than min");
    minvalue = static_cast<scalar_t>(range.value()[0]);
    maxvalue = static_cast<scalar_t>(range.value()[1]);
    // If the values in range cannot be represented by input dtype, we avoid
    // promoting the tensor and instead output a warning.
    if (static_cast<double>(minvalue) != range.value()[0] ||
        static_cast<double>(maxvalue) != range.value()[1]) {
      TORCH_WARN_ONCE(
          "Value in range cannot be represented by tensor's scalar type, casting to ",
          self.scalar_type());
    }
  } else {
    minvalue = *self.min().cpu().data_ptr<scalar_t>();
    maxvalue = *self.max().cpu().data_ptr<scalar_t>();
    // This is done to avoid divide by zero if input min is equal to input max.
    // In this case computing the histogram can also be skipped altogether, as
    // it's equal to the sum of weights in the middle bin, and zero everywhere
    // else.
    if (minvalue == maxvalue) {
      minvalue -= 1;
      maxvalue += 1;
    }
  }
  TORCH_CHECK(
      !(std::isinf(static_cast<double>(minvalue)) ||
        std::isinf(static_cast<double>(maxvalue)) || _isnan(minvalue) ||
        _isnan(maxvalue)),
      "range of [",
      minvalue,
      ", ",
      maxvalue,
      "] is not finite");
  return std::make_pair(minvalue, maxvalue);
}

Tensor _histogram_normalize_density(const Tensor& self, const Tensor& edges, bool uniform_bins) {
  auto scalar_target = c10::get_default_dtype();
  if (self.is_floating_point()) {
    scalar_target = self.scalar_type();
  }
  Tensor output = self.to(scalar_target);
  if (uniform_bins) {
    int64_t nbins = self.size(0);
    double bin_volume =
        (edges[nbins].item<double>() - edges[0].item<double>()) /
        static_cast<double>(nbins);
    output /= bin_volume * output.sum();
  } else {
    output /= output.sum() *
        (edges.slice(0, 1, edges.numel()) - edges.slice(0, 0, -1))
            .to(scalar_target);  
  }
  return output;
}

}} //at::native