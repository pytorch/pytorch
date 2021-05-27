#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

#include <tuple>
#include "c10/core/DefaultDtype.h"

/* Implement a numpy-like histogram function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * - torch.histogram(input, bins, range=None, weight=None, density=False)
 *   input     - tensor containing the input values. The histogram is computed over the flattened values.
 *   bins      - int or 1D tensor. If int, defines the number of equal-width bins. If tensor, defines the
 *               sequence of bin edges including the rightmost edge.
 *   min       - scalar, optional. Defines the lower range of the bins. If not provided, defaults to input.min().
 *   max       - scalar, optional. Defines the upper range of the bins. If not provided, defaults to input.max().
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
 * Restrictions are defined in histogram_pre_check()
 */

namespace at { namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_stub);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(histogram_linear_stub);

namespace {

/* Shape and dtype checks for histogram's input tensors (input, weight)
 * and output tensors (hist, bin_edges).
 */
inline void histogram_pre_check(const Tensor& input, const Tensor& bin_edges_in,
        const c10::optional<Tensor>& weight, bool density,
        const Tensor& hist, const Tensor& bin_edges_out) {
    TORCH_CHECK(input.dtype() == bin_edges_in.dtype(), "torch.histogram(): input tensor and bins tensor should",
            " have same dtype, but got input ", input.dtype(), " and bins ", bin_edges_in.dtype());

    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram(): input tensor and hist tensor should",
            " have same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    TORCH_CHECK(input.dtype() == bin_edges_out.dtype(), "torch.histogram(): input tensor and bin_edges tensor should",
            " have same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges_out.dtype());

    TORCH_CHECK(hist.dim() == 1, "torch.histogram(): hist tensor should have dimension 1,",
            " but got ", hist.dim(), " dimension");

    TORCH_CHECK(bin_edges_in.dim() == 1, "torch.histogram(): bins tensor should have dimension 1,",
            " but got ", bin_edges_in.dim(), " dimension");

    TORCH_CHECK(bin_edges_out.dim() == 1, "torch.histogram(): bin_edges tensor should have dimension 1,",
            " but got ", bin_edges_out.dim(), " dimension");

    if (weight.has_value()) {
        auto weight_sizes = weight.value().sizes();
        TORCH_CHECK(input.sizes() == weight.value().sizes(), "torch.histogram(): if weight tensor is provided,"
                " input tensor and weight tensor should have the same shape, but got input(", input.sizes(), ")",
                ", and weight(", weight_sizes, ")");
    }

    TORCH_CHECK(hist.is_contiguous(), "torch.histogram(): hist tensor must be contiguous");

    TORCH_CHECK(bin_edges_in.numel() > 0, "torch.histogram(): bin_edges tensor should have at least 1 element,",
            " but got ", bin_edges_in.numel(), " elements");

    at::native::resize_output(hist, {bin_edges_in.numel() - 1});

    at::native::resize_output(bin_edges_out, {bin_edges_in.numel()});
}

/* Determines the outermost bin edges.
 */
std::pair<double, double> select_outer_bin_edges(const Tensor& input,
        const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
    if (min.has_value() && max.has_value()) {
        return std::make_pair(min.value().to<double>(), max.value().to<double>());
    }

    // Default range for empty input matching numpy.histogram's default
    if (input.numel() == 0) {
        return std::make_pair(0., 1.);
    }

    auto extrema = _aminmax(input);

    double leftmost_edge  = min.has_value() ? min.value().to<double>() : std::get<0>(extrema).item<double>();
    double rightmost_edge = max.has_value() ? max.value().to<double>() : std::get<1>(extrema).item<double>();

    TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogram(): min should not exceed max, but got",
            "min ", leftmost_edge, " max ", rightmost_edge);

    // Expand empty range to match numpy behavior and avoid division by 0 in normalization
    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 0.5;
        rightmost_edge += 0.5;
    }

    return std::make_pair(leftmost_edge, rightmost_edge);
}

} // namespace

std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, const Tensor& bin_edges,
        const c10::optional<Scalar>& min, const c10::optional<Scalar>& max,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges_out) {
    histogram_pre_check(self, bin_edges, weight, density, hist, bin_edges_out);
    bin_edges_out.copy_(bin_edges);
    histogram_stub(self.device().type(), self, weight, density, hist, bin_edges_out);
    return std::forward_as_tuple(hist, bin_edges_out);
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, const Tensor& bin_edges,
        const c10::optional<Scalar>& min, const c10::optional<Scalar>& max,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges_out = at::empty({0}, bin_edges.options());
    return histogram_out_cpu(self, bin_edges, min, max, weight, density, hist, bin_edges_out);
}

std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, int64_t bin_ct,
        const c10::optional<Scalar>& min, const c10::optional<Scalar>& max,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges_out) {
    auto outer_bin_edges = select_outer_bin_edges(self, min, max);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1, bin_edges_out);

    histogram_pre_check(self, bin_edges_out, weight, density, hist, bin_edges_out);
    histogram_linear_stub(self.device().type(), self, weight, density, hist, bin_edges_out);
    return std::forward_as_tuple(hist, bin_edges_out);
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, int64_t bin_ct,
        const c10::optional<Scalar>& min, const c10::optional<Scalar>& max,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges_out = at::empty({0}, self.options());
    return histogram_out_cpu(self, bin_ct, min, max, weight, density, hist, bin_edges_out);
}

}} // namespace at::native
