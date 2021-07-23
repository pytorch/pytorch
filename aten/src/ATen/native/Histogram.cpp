#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

#include <tuple>
#include <c10/core/ScalarType.h>
#include <c10/core/DefaultDtype.h>

/* Implements a numpy-like histogram function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * - torch.histogram(input, bins, range=None, weight=None, density=False)
 *   input     - tensor containing the input values. The histogram is computed over the flattened values.
 *   bins      - int or 1D tensor. If int, defines the number of equal-width bins. If tensor, defines the
 *               sequence of bin edges including the rightmost edge.
 *   range     - (float, float), optional. Defines the range of the bins.
 *   weight    - tensor, optional. If provided, weight should have the same shape as input. Each value
 *               in input contributes its associated weight towards its bin's result (instead of 1).
 *   density   - bool, optional. If False, the result will contain the number of samples (or total weight)
 *               in each bin. If True, the result is the value of the probability density function at the
 *               bin, normalized such that the integral over the range is 1.
 *
 * Returns:
 *   hist      - 1D tensor containing the values of the histogram.
 *   bin_edges - 1D tensor containing the edges of the histogram bins. Contains hist.numel() + 1 elements.
 *               Bins include their left edge and exclude their right edge, with the exception of the
 *               rightmost bin which includes both of its edges.
 *
 * Restrictions are defined in histogram_check_inputs() and in select_outer_bin_edges().
 */

namespace at { namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_stub);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_linear_stub);

namespace {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogram_check_inputs(const Tensor& input, const Tensor& bins, const c10::optional<Tensor>& weight) {
    TORCH_CHECK(input.dtype() == bins.dtype(), "torch.histogram: input tensor and bins tensor should",
            " have the same dtype, but got input ", input.dtype(), " and bins ", bins.dtype());

    TORCH_CHECK(bins.dim() == 1, "torch.histogram: bins tensor should have dimension 1,",
            " but got ", bins.dim(), " dimension");

    TORCH_CHECK(bins.numel() > 0, "torch.histogram: bins tensor should have at least 1 element,",
            " but got ", bins.numel(), " elements");

    if (weight.has_value()) {
        TORCH_CHECK(input.dtype() == weight.value().dtype(), "torch.histogram: if weight tensor is provided,"
                " input tensor and weight tensor should have the same dtype, but got input(", input.dtype(), ")",
                ", and weight(", weight.value().dtype(), ")");

        TORCH_CHECK(input.sizes() == weight.value().sizes(), "torch.histogram: if weight tensor is provided,"
                " input tensor and weight tensor should have the same shape, but got input(", input.sizes(), ")",
                ", and weight(", weight.value().sizes(), ")");
    }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogram_prepare_out(const Tensor& input, int64_t bin_ct,
        const Tensor& hist, const Tensor& bin_edges) {
    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram: input tensor and hist tensor should",
            " have the same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    TORCH_CHECK(input.dtype() == bin_edges.dtype(), "torch.histogram: input tensor and bin_edges tensor should",
            " have the same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges.dtype());

    TORCH_CHECK(bin_ct > 0,
            "torch.histogram(): bins must be > 0, but got ", bin_ct);

    at::native::resize_output(hist, bin_ct);

    at::native::resize_output(bin_edges, bin_ct + 1);

    TORCH_CHECK(hist.is_contiguous(), "torch.histogram: hist tensor must be contiguous");
}

/* Determines the outermost bin edges.
 */
std::pair<double, double> select_outer_bin_edges(const Tensor& input, c10::optional<c10::ArrayRef<double>> range) {
    TORCH_CHECK(!range.has_value() || range.value().size() == 2, "torch.histogram: range should have 2 elements",
            " if specified, but got ", range.value().size());

    // Default range for empty input matching numpy.histogram's default
    double leftmost_edge = 0., rightmost_edge = 1.;

    if (range.has_value()) {
        // range is specified
        leftmost_edge = range.value()[0];
        rightmost_edge = range.value()[1];
    } else if (input.numel() > 0) {
        // non-empty input
        auto extrema = _aminmax(input);
        leftmost_edge = std::get<0>(extrema).item<double>();
        rightmost_edge = std::get<1>(extrema).item<double>();
    }

    TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
            std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "torch.histogram: range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite");

    TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogram: min should not exceed max, but got",
            " min ", leftmost_edge, " max ", rightmost_edge);

    // Expand empty range to match numpy behavior and avoid division by 0 in normalization
    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 0.5;
        rightmost_edge += 0.5;
    }

    return std::make_pair(leftmost_edge, rightmost_edge);
}

/* histc's version of the logic for outermost bin edges.
 */
std::pair<double, double> histc_select_outer_bin_edges(const Tensor& input,
        const Scalar& min, const Scalar& max) {
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();

    if (leftmost_edge == rightmost_edge) {
        auto extrema = _aminmax(input);
        leftmost_edge = std::get<0>(extrema).item<double>();
        rightmost_edge = std::get<1>(extrema).item<double>();
    }

    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 1;
        rightmost_edge += 1;
    }

    TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
            std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "torch.histc: range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite");

    TORCH_CHECK(leftmost_edge < rightmost_edge, "torch.histc: max must be larger than min");

    return std::make_pair(leftmost_edge, rightmost_edge);
}

} // namespace

/* Versions of histogram in which bins is a Tensor defining the sequence of bin edges.
 */
std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, const Tensor& bins,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    histogram_check_inputs(self, bins, weight);
    histogram_prepare_out(self, bins.numel() - 1, hist, bin_edges);

    bin_edges.copy_(bins);
    histogram_stub(self.device().type(), self, weight, density, hist, bin_edges);
    return std::forward_as_tuple(hist, bin_edges);
}
std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, const Tensor& bins,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);
    return histogram_out_cpu(self, bins, weight, density, hist, bin_edges);
}

/* Versions of histogram in which bins is an integer specifying the number of equal-width bins.
 */
std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, int64_t bin_ct, c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    histogram_prepare_out(self, bin_ct, hist, bin_edges);
    auto outer_bin_edges = select_outer_bin_edges(self, range);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1, bin_edges);
    histogram_check_inputs(self, bin_edges, weight);

    histogram_linear_stub(self.device().type(), self, weight, density, hist, bin_edges, true);
    return std::forward_as_tuple(hist, bin_edges);
}
std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, int64_t bin_ct, c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges_out = at::empty({0}, self.options());
    return histogram_out_cpu(self, bin_ct, range, weight, density, hist, bin_edges_out);
}

/* Narrowed interface for the legacy torch.histc function.
 */
Tensor& histogram_histc_cpu_out(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max, Tensor& hist) {
    Tensor bin_edges = at::empty({0}, self.options());
    histogram_prepare_out(self, bin_ct, hist, bin_edges);
    auto outer_bin_edges = histc_select_outer_bin_edges(self, min, max);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1, bin_edges);
    histogram_check_inputs(self, bin_edges, {});

    histogram_linear_stub(self.device().type(), self,
            c10::optional<Tensor>(), false, hist, bin_edges, false);
    return hist;
}
Tensor histogram_histc_cpu(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    return histogram_histc_cpu_out(self, bin_ct, min, max, hist);
}

}} // namespace at::native
