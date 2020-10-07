// Returns the frequency of elements of input non-negative integer tensor.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>

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

// A helper function to get the bin for histogram
// while each bin is inclusive at the lower end and exclusive at the higher,
// i.e. [start, end) the last bin is inclusive at both, i.e. [start, end], in
// order to include maxvalue if exists
template <typename input_t>
inline int64_t getbin(input_t x, input_t min, input_t max, int64_t bins) {
  if (x == max)
    return bins - 1;
  return static_cast<int64_t>((x - min) * bins / (max - min));
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
  // This is done to avoid divide by zero. Technically, it results in incorrect behavior in that case,
  // but it is consistent with numpy.
  if (min == max) {
    min -= .5;
    max += .5;
  }
  TORCH_CHECK(nbins > 0, "bins must be > 0");
  TORCH_CHECK(
      !(std::isinf(static_cast<double>(min)) ||
        std::isinf(static_cast<double>(max)) ||
        _isnan(min) || _isnan(max)),
      "range of [",
      min,
      ", ",
      max,
      "] is not finite");
  TORCH_CHECK(min < max, "max must be larger than min");

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }
  Tensor output;
  int64_t self_size = self.size(0);

  const input_t* self_p = self.data_ptr<input_t>();
  if (has_weights) {
    output = native::zeros({nbins}, weights.options());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.data_ptr<weights_t>();
    // This does the actual work of computing the histogram.
    // This is single threaded, as other similar operators are single-threaded in PyTorch today, and a multi-threaded one would be tricky to implement.
    for (int64_t i = 0; i < self_size; i++) {
      if (self_p[i] >= min && self_p[i] <= max)
        output_p[getbin(self_p[i], min, max, nbins)] += weights_p[i];
    }
  } else {
    output = native::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    for (int64_t i = 0; i < self_size; i++) {
      if (self_p[i] >= min && self_p[i] <= max)
        output_p[getbin(self_p[i], min, max, nbins)] += 1L;
    }
  }

  return output;

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


std::tuple<Tensor,Tensor> _histogram_cpu(
    const Tensor& self,
    int64_t bins,
    c10::optional<ArrayRef<double>> range,
    const Tensor& weights,
    bool density) {

  // Weights having a different shape from input is not supported yet. TO DO:
  // Add support for weights broadcastable to input
  bool has_weights = weights.defined();
  Tensor flattened_weights;
  if (has_weights) {
    TORCH_CHECK(
        weights.sizes() == self.sizes(),
        "histogram only supports input and weights of the same shape");
    flattened_weights = weights.flatten(0).contiguous();
  }


  // If range is not defined, compute min and max from the values in the Tensor
  Scalar min;
  Scalar max;
  if (range.has_value()) {
    min = range.value()[0];
    max = range.value()[1];
  } else {
    min = self.min().item();
    max = self.max().item();
  }

  Tensor hist = AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "histogram_cpu_uniformbins", [&] {
        const auto scalar = weights.scalar_type();
        if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
          return _histogram_cpu_template_uniformbins<scalar_t, float>(
              self.flatten(0).contiguous(),
              bins,
              flattened_weights,
              min.to<scalar_t>(),
              max.to<scalar_t>(),
              density);
        return _histogram_cpu_template_uniformbins<scalar_t, double>(
            self.flatten(0).contiguous(),
            bins,
            flattened_weights.to(kDouble),
            min.to<scalar_t>(),
            max.to<scalar_t>(),
            density);
      });

  if (density) { // Compute the density
    double minval = min.to<double>();
    double maxval = max.to<double>();
    if (minval == maxval) {
      minval -= 1;
      maxval += 1;
    }
    hist = hist.to(kDouble);
    hist *= bins / (maxval - minval) / hist.sum();
  }

  Tensor edges;
  if (self.scalar_type() == kFloat) {
    edges = at::linspace(min, max, bins + 1, self.options());
  } else {
    edges = at::linspace(min, max, bins + 1, self.options().dtype(kDouble));
  }

  return std::make_tuple(hist, edges);
}

std::tuple<Tensor, Tensor> histogram(
    const Tensor& self,
    const Tensor& bins,
    const Tensor& weights,
    bool density) {

  TORCH_CHECK(bins.dim() == 1, "bins must be 1d, when a tensor");

  //Skip the input check for CUDA to avoid device synchronization.
  if (self.device().type() == kCPU) {
    TORCH_CHECK(
        at::all(bins.slice(0, 1, bins.numel()) >= bins.slice(0, 0, -1))
            .item<bool>(),
        "bin edges must increase monotonically"); 
  }

  Tensor flattened_weights;
  if (weights.defined()) {
    TORCH_CHECK(
        weights.sizes() == self.sizes(),
        "histogram only supports input and weights of the same shape");
    flattened_weights = weights.flatten(0).contiguous();
  }
  int64_t nbins = bins.size(0) - 1;
  Tensor index = searchsorted(bins, self, false, true);
  index.clamp_(0, nbins + 2);
  index = index.where(self != bins[-1], index - 1);
  Tensor hist = bincount(index.flatten(0), flattened_weights, nbins + 2);

  hist = hist.slice(0, 1, -1);

  if (density) { // Compute the density
    hist = hist.to(kDouble);
    hist /= hist.sum() *
        (bins.slice(0, 1, bins.numel()) - bins.slice(0, 0, -1)).to(kDouble);
    }

  return std::make_tuple(hist, bins);
}


}} // namespace at::native
