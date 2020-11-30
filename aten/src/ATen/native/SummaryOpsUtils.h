#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

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