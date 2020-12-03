// Returns the frequency of elements of input non-negative integer tensor.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/SummaryOpsUtils.h>

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
inline int64_t getbin(input_t x, input_t min, input_t max, int64_t nbins) {
  if (x >= max) {
    return nbins - 1;
  }
  return static_cast<int64_t>((x - min) * nbins / (max - min));
}

///////////////// histogram /////////////////
// This template assumes that input and weights are contiguous (possibly
// flattened) 1D Tensors of the same size. This can be fixed later on.
template <typename input_t, typename weights_t>
std::tuple<Tensor, Tensor> _histogram_cpu_template_uniform_bins(
    const Tensor& self,
    int64_t nbins,
    const Tensor& weights /* optional */,
    c10::optional<ArrayRef<double>> range,
    bool density) {

  auto computed_range = _histogram_maybe_compute_range<input_t>(self, range);
  input_t minvalue = computed_range.first;
  input_t maxvalue = computed_range.second;

  TORCH_CHECK(nbins > 0, "bins must be > 0");

  bool has_weights = weights.defined();

  Tensor hist;
  int64_t self_size = self.size(0);

  const input_t* self_p = self.data_ptr<input_t>();
  if (has_weights) {
    hist = native::zeros({nbins}, weights.options());
    weights_t* output_p = hist.data_ptr<weights_t>();
    const weights_t* weights_p = weights.data_ptr<weights_t>();
    // This does the actual work of computing the histogram.
    // This is single threaded, as other similar operators are single-threaded
    // in PyTorch today, and a multi-threaded one would be tricky to implement.
    // TODO: Implement a multithreaded version.
    for (int64_t i = 0; i < self_size; i++) {
      if (self_p[i] >= minvalue && self_p[i] <= maxvalue) {
        output_p[getbin(self_p[i], minvalue, maxvalue, nbins)] += weights_p[i];
      }
    }
  } else {
    hist = native::zeros({nbins}, kLong);
    int64_t* output_p = hist.data_ptr<int64_t>();
    for (int64_t i = 0; i < self_size; i++) {
      if (self_p[i] >= minvalue && self_p[i] <= maxvalue) {
        output_p[getbin(self_p[i], minvalue, maxvalue, nbins)] += static_cast<int64_t>(1);
      }
    }
  }

  Tensor edges;
  if (c10::isFloatingType(self.scalar_type())) {
    edges = at::linspace(minvalue, maxvalue, nbins + 1, self.options());
  } else {
    edges = at::linspace(
        minvalue, maxvalue, nbins + 1, self.options().dtype(c10::get_default_dtype()));
  }
  
  if (density) { // Compute the density
    hist = _histogram_normalize_density(hist, edges, true);
  }

  return std::make_tuple(hist, edges);

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

std::tuple<Tensor,Tensor> _histogram_cpu_uniform_bins(
    const Tensor& self,
    int64_t nbins,
    c10::optional<ArrayRef<double>> range,
    const Tensor& weights /* optional */,
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

  // There might be a better way for dispatching over pairs of dtypes.
  return AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "histogram_cpu_uniform_bins", [&] {
    const auto scalar = weights.scalar_type();
        switch (scalar) {
          case ScalarType::Float:
            return _histogram_cpu_template_uniform_bins<scalar_t, float>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Double:
            return _histogram_cpu_template_uniform_bins<scalar_t, double>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Byte:
            return _histogram_cpu_template_uniform_bins<scalar_t, uint8_t>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Char:
            return _histogram_cpu_template_uniform_bins<scalar_t, int8_t>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Short:
            return _histogram_cpu_template_uniform_bins<scalar_t, int16_t>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Int:
            return _histogram_cpu_template_uniform_bins<scalar_t, int32_t>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          case ScalarType::Long:
          case ScalarType::Undefined:
            return _histogram_cpu_template_uniform_bins<scalar_t, int64_t>(
                self.flatten(0).contiguous(),
                nbins,
                flattened_weights,
                range,
                density);
          default:
            TORCH_CHECK(
                false, "Scalar type ", scalar, " not supported for weights");
            return std::make_tuple(Tensor(), Tensor());
        }
    });

}

// Device-generic implementation for histogram with custom, possibly non-uniform binning
std::tuple<Tensor, Tensor> histogram(
    const Tensor& self,
    const Tensor& bins,
    const Tensor& weights,
    bool density) {

  TORCH_CHECK(
      bins.dim() == 1,
      "custom bin edges must be represented as a one dimensional tensor, but got a tensor with dimension ",
      bins.dim());

  // Skip the input check for CUDA to avoid device synchronization.
  if (self.device().type() == kCPU) {
    TORCH_CHECK(
        at::all(bins.slice(0, 1, bins.numel()) >= bins.slice(0, 0, -1))
            .item<bool>(),
        "bin edges must increase monotonically"); 
  }

  // Flatten the weights as bincount only accepts weights as a 1D Tensor.
  Tensor flattened_weights;
  if (weights.defined()) {
    TORCH_CHECK(
        weights.sizes() == self.sizes(),
        "histogram only supports input and weights of the same shape");
    // Throw an error if weights scalar type is complex, as bincount casts them
    // to double. TO DO: Add a workaround for that.
    TORCH_CHECK(
        !isComplexType(weights.scalar_type()),
        "Scalar type ",
        weights.scalar_type(),
        " not supported for weights");
    flattened_weights = weights.flatten(0).contiguous();
  }

  int64_t nbins = bins.size(0) - 1;
  // Perform the bin search
  Tensor index = searchsorted(bins, self, false, true);
  // Make the last bin inclusive
  index = index.where(self != bins[-1], index - 1);
  // Compute the histogram - nbins+2 because of also counting in the overflow bins.
  Tensor hist = bincount(index.flatten(0), flattened_weights, nbins + 2);
  // Remove the overflow bins
  hist = hist.slice(0, 1, -1);

  // Compute the density
  if (density) { 
    // Hack for float32 default dtype and integral weights because bincount promotes to double
    if (weights.defined())
      hist = hist.to(weights.scalar_type());
    hist = _histogram_normalize_density(hist, bins, false);
  } else {
    // This is a hack for integral types of weights, as bincount casts them to double when computing the histogram
    // and disabling that would be a bc-breaking change
    if (weights.defined())
      hist = hist.to(weights.scalar_type());
  }

  return std::make_tuple(hist, bins);
}


}} // namespace at::native
