#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Histogram.h>

#include <tuple>

namespace at { namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_stub);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_linear_stub);

namespace {

inline void histogram_pre_check(const Tensor& input, const Tensor& bins, const c10::optional<Tensor>& weight,
    bool density, const Tensor& hist, const Tensor& bin_edges) {
  TORCH_CHECK(bins.device() == input.device(), "bins and input value tensors should have same device type, ",
    "but we got bins tensor device type ", bins.device(), " and input value tensor device type ", input.device());

  TORCH_CHECK(bins.dim() == 1, "bins tensor should have dimension 1, but got ", bins.dim(), " dimension");

  TORCH_CHECK(hist.dim() == 1, "hist tensor should have dimension 1, but got ", hist.dim(), " dimension");

  TORCH_CHECK(bin_edges.dim() == 1, "bin_edges tensor should have dimension 1, but got ", bin_edges.dim(), " dimension");

  TORCH_CHECK(bins.numel() > 0, "bins tensor should have at least 1 element, but has 0");

  if (weight.has_value()) {
    auto weight_sizes = weight.value().sizes();
    TORCH_CHECK(input.sizes() == weight.value().sizes(), "if weight tensor is provided, input value tensor and weight tensor",
        " should have the same shape, but we got input(", input.sizes(), ") and weight(", weight_sizes, ")");

    ScalarType weight_dtype = weight.value().scalar_type();
    TORCH_CHECK(weight_dtype == ScalarType::Float || weight_dtype == ScalarType::Double,
        "weight tensor's dtype must be Float or Double, but we got weight tensor's type ", weight_dtype);
  }

  TORCH_CHECK(hist.is_contiguous(), "hist output tensor must be contiguous");

  TORCH_CHECK(bin_edges.is_contiguous(), "bin_edges output tensor must be contiguous");

  ScalarType hist_dtype = hist.scalar_type();
  if (density) {
    TORCH_CHECK(hist_dtype == ScalarType::Double, "hist tensor's dtype is wrong, it must be Double if density flag is True, "
        "but we got hist tensor's type ", hist_dtype, " and density flag is ", (density ? "True" : "False"));
  } else if (weight.has_value()) {
    TORCH_CHECK(hist_dtype == ScalarType::Double, "hist tensor's dtype is wrong, it must be Double if weight is provided, "
        "but we got hist tensor's type ", hist_dtype);
  } else {
    TORCH_CHECK(hist_dtype == ScalarType::Long, "hist tensor's dtype is wrong, it should be long in the unweighted case ",
        "but we got hist tensor's type ", hist_dtype);
  }
}

ScalarType hist_scalar_type(const c10::optional<Tensor>& weight, bool density) {
    if (density || weight.has_value()) {
        return ScalarType::Double;
    }
    return ScalarType::Long;
}

std::pair<double, double> select_outer_bin_edges(const Tensor& input, const Scalar& min, const Scalar& max) {
    double min_val = min.to<double>();
    double max_val = max.to<double>();

    TORCH_CHECK(min_val <= max_val, "min should not exceed max, but we got min=", min_val, " max=", max_val,
            " (after conversion to double precision)");

    if (min_val != max_val) {
        return {min_val, max_val};
    }

    if (input.numel() == 0) {
        return {0., 1.};
    }

    double first_edge = input.min().item<double>();
    double last_edge = input.max().item<double>();

    if (first_edge == last_edge) {
        first_edge -= 0.5;
        last_edge += 0.5;
    }

    return {first_edge, last_edge};
}

} // namespace

std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, const Tensor& bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    histogram_pre_check(self, bins, weight, density, hist, bin_edges);
    histogram_stub(self.device().type(), self, bins, min, max, weight, density, hist, bin_edges);

    return {hist, bin_edges};
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, const Tensor& bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density) {
    c10::TensorOptions hist_options = TensorOptions().device(self.options().device()).dtype(hist_scalar_type(weight, density));
    Tensor hist = at::empty({0}, hist_options, MemoryFormat::Contiguous);
    Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);

    return histogram_out_cpu(self, bins, min, max, weight, density, hist, bin_edges);
}

std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    auto outer_bin_edges = select_outer_bin_edges(self, min, max);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bins + 1, bin_edges);

    histogram_pre_check(self, bin_edges, weight, density, hist, bin_edges);
    histogram_linear_stub(self.device().type(), self, bin_edges, min, max, weight, density, hist, bin_edges);

    return {hist, bin_edges};
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density) {
    c10::TensorOptions hist_options = TensorOptions().device(self.options().device()).dtype(hist_scalar_type(weight, density));
    Tensor hist = at::empty({0}, hist_options, MemoryFormat::Contiguous);

    c10::TensorOptions bin_edges_options = TensorOptions().device(self.options().device()).dtype(ScalarType::Double);
    Tensor bin_edges = at::empty({0}, bin_edges_options, MemoryFormat::Contiguous);

    return histogram_out_cpu(self, bins, min, max, weight, density, hist, bin_edges);
}

}} // namespace at::native
