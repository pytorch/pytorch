// Returns the frequency of elements of input non-negative integer tensor.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <tuple>

namespace at { namespace native {

namespace {
///////////////// bincount /////////////////
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
    output = native::zeros({nbins}, weights.options());
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

///////////////// histogram /////////////////
template <typename input_t, typename weights_t>
Tensor _histogram_cpu_template_uniformbins(
    const Tensor& self,
    int64_t nbins,
    const Tensor& weights,
    input_t min,
    input_t max,
    bool density) {
  input_t minval = min;
  input_t maxval = max;
  if (min == max) {
    minval = self.min().item<input_t>();
    maxval = self.max().item<input_t>();
  }
  if (minval == maxval) {
    minval -= 1.0;
    maxval += 1.0;
  }
  TORCH_CHECK(nbins > 0, "bins must be > 0");
  // Check if range is finite, hack to keep MSVC from complaining
  TORCH_CHECK(
      !(std::isinf((double)minval) || std::isinf((double)maxval) ||
        std::isnan((double)minval) || std::isnan((double)maxval)),
      "range of [",
      minval,
      ", ",
      maxval,
      "] is not finite"); 
  TORCH_CHECK(minval < maxval, "max must be larger than min");
  input_t scale = (input_t)nbins / (maxval - minval);
  Tensor index = ((self - minval) * scale+1).to(kLong);
  index = index.where(self != maxval, index-1); //The uppermost bin is closed at both sides
  index.clamp_(0, nbins + 1);
  Tensor hist =
      _bincount_cpu_template<int64_t,weights_t>(index, weights, nbins + 1); // A workaround, since histc doesn't take
                                 // weights, we'll instead use bincount

  hist = hist.slice(0, 1, nbins+1);

  return hist;

}

template <typename input_t, typename weights_t>
Tensor _histogram_cpu_template_custombins(
    const Tensor& self,
    const Tensor& bins,
    const Tensor& weights,
    bool density) {
  TORCH_CHECK(bins.dim() == 1, "bins must be 1d, when a tensor");
  TORCH_CHECK(
      at::all(bins.slice(0, 1, bins.numel()) >= bins.slice(0, 0, -1)).item<bool>(),
      "bins must increase monotonically");
  input_t maxval = bins[-1].item<input_t>();
  int64_t nbins = bins.numel() - 1L;
  Tensor index = searchsorted_cpu(bins,self,false,true);
  index.clamp_(0, nbins + 2L);
  index = index.where(self != maxval, index - 1L);
  Tensor hist = _bincount_cpu_template<int64_t, weights_t>(
      index, weights, nbins + 2L); 

  hist = hist.slice(0, 1, -1);

  return hist;
}

} // namespace

Tensor _bincount_cpu(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self.contiguous(), weights.contiguous(), minlength);
    return _bincount_cpu_template<scalar_t, double>(
        self.contiguous(), weights.contiguous().to(kDouble), minlength);
  });
}


Tensor _histogram_cpu(
    const Tensor& self,
    int64_t bins,
    const Tensor& weights,
    Scalar min,
    Scalar max,
    bool density) {

  auto hist = AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histogram_cpu_uniformbins", [&] {
    const auto scalar = weights.scalar_type();
        if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
          return _histogram_cpu_template_uniformbins<scalar_t, float>(
              self.flatten(0).contiguous(),
              bins,
              weights.contiguous(),
              min.to<scalar_t>(),
              max.to<scalar_t>(),
              density);
        return _histogram_cpu_template_uniformbins<scalar_t, double>(
            self.flatten(0).contiguous(),
            bins,
            weights.contiguous().to(kDouble),
            min.to<scalar_t>(),
            max.to<scalar_t>(),
            density);
  });

  
  if (density) { // Compute the density
    double minval = min.to<double>();
    double maxval = max.to<double>();
    if (minval == maxval) {
      minval = self.min().to(kDouble).item<double>();
      maxval = self.max().to(kDouble).item<double>();
    }
    if (minval == maxval) {
      minval -= 1.0;
      maxval += 1.0;
    }
    hist = hist.to(kDouble);
    hist *= bins / (maxval - minval) / hist.sum();
  }

  return hist;
}

Tensor _histogram_cpu(
    const Tensor& self,
    const Tensor& bins,
    const Tensor& weights,
    bool density) {

    auto hist = AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "histogram_cpu_custombins", [&] {
        const auto scalar = weights.scalar_type();
        if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
          return _histogram_cpu_template_custombins<scalar_t, float>(
              self.flatten(0).contiguous(),
              bins,
              weights.contiguous(),
              density);
        return _histogram_cpu_template_custombins<scalar_t, double>(
            self.flatten(0).contiguous(),
            bins,
            weights.contiguous().to(kDouble),
            density);
      });

  if (density) { // Compute the density
    hist = hist.to(kDouble);
    hist /= hist.sum() *
        (bins.slice(0, 1, bins.numel()) - bins.slice(0, 0, -1)).to(kDouble);
    }

  return hist;
}


}} // namespace at::native
