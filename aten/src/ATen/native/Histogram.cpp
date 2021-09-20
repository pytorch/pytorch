#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

#include <tuple>
#include <vector>
#include <c10/core/ScalarType.h>
#include <c10/core/DefaultDtype.h>

/* Implements a numpy-like histogramdd function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 *
 * - torch.histogramdd(input, bins, range=None, weight=None, density=False)
 *   input     - tensor with dimensions (N, D), interpreted as N coordinates in a D-dimensional space.
 *               If a tensor with more than 2 dimensions is passed, all but the last dimension will be flattened.
 *   bins      - int[] of length D or tensor list of length D. If int[], defines the number of equal-width bins
 *               in each dimension. If tensor list, defines the sequences of bin edges, including rightmost edges,
 *               for each dimension.
 *   range     - float[] of length D, optional. If specified, defines the leftmost and rightmost bin edges
 *               for each dimension.
 *   weight    - tensor, optional. If provided, weight should have the same shape as input excluding its last dimension.
 *               Each D-dimensional coordinate in input contributes its associated weight towards its bin's result.
 *               If weight is not specified, each coordinate has weight 1 by default.
 *   density   - bool, optional. If False, the result will contain the number of samples (or total weight)
 *               in each bin. If True, the result is the value of the probability density function at the
 *               bin, normalized such that the integral over all bins is 1.
 *
 * Returns:
 *   hist      - D-dimensional tensor containing the values of the histogram.
 *   bin_edges - tensor list of length D containing the edges of the histogram bins in each dimension.
 *               Bins include their left edge and exclude their right edge, with the exception of the
 *               rightmost bin in each dimension which includes both of its edges.
 *
 * Restrictions are defined in histogram_check_inputs() and in select_outer_bin_edges().
 */

namespace at { namespace native {

DEFINE_DISPATCH(histogramdd_stub);
DEFINE_DISPATCH(histogram_linear_stub);

namespace {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogramdd_check_inputs(const Tensor& input, const TensorList& bins, const c10::optional<Tensor>& weight) {
    TORCH_CHECK(input.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions");

    const int64_t D = input.size(-1);

    for (int64_t dim = 0; dim < D; dim++) {
        TORCH_CHECK(input.dtype() == bins[dim].dtype(), "torch.histogramdd: input tensor and bins tensor should",
                " have the same dtype, but got input ", input.dtype(),
                " and bins for dimension ", dim, bins[dim].dtype());

        TORCH_CHECK(bins[dim].dim() == 1, "torch.histogramdd: bins tensor should have dimension 1,",
                " but got ", bins[dim].dim(), " dimensions for dimension ", dim);

        TORCH_CHECK(bins[dim].numel() > 0, "torch.histogramdd: bins tensor should have at least 1 element,",
                " but got ", bins[dim].numel(), " elements for dimension ", dim);
    }

    if (weight.has_value()) {
        TORCH_CHECK(input.dtype() == weight.value().dtype(), "torch.histogram: if weight tensor is provided,"
                " input tensor and weight tensor should have the same dtype, but got input(", input.dtype(), ")",
                ", and weight(", weight.value().dtype(), ")");

        auto input_sizes = input.sizes().vec();
        input_sizes.pop_back();

        auto weight_sizes = weight.value().sizes().vec();
        if (weight_sizes.empty())
            weight_sizes = {1}; // correctly handle scalars

        TORCH_CHECK(input_sizes == weight_sizes, "torch.histogram: if weight tensor is provided,"
                " it should have the same shape as the input tensor excluding its last dimension, but got input ",
                input.sizes(), " and weight ", weight.value().sizes());
    }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogramdd_prepare_out(const Tensor& input, const std::vector<int64_t>& bin_ct,
        const Tensor& hist, const TensorList& bin_edges) {
    const int64_t D = input.size(-1);

    TORCH_INTERNAL_ASSERT(int64_t(bin_ct.size()) == D);
    TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);

    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram: input tensor and hist tensor should",
            " have the same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    for (int64_t dim = 0; dim < D; dim++) {
        TORCH_CHECK(input.dtype() == bin_edges[dim].dtype(), "torch.histogram: input tensor and bin_edges tensor should",
                " have the same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges[dim].dtype(),
                " for dimension ", dim);

        TORCH_CHECK(bin_ct[dim] > 0,
                "torch.histogram(): bins must be > 0, but got ", bin_ct[dim], " for dimension ", dim);

        at::native::resize_output(bin_edges[dim], bin_ct[dim] + 1);
    }

    at::native::resize_output(hist, bin_ct);
}
void histogramdd_prepare_out(const Tensor& input, TensorList bins,
        const Tensor& hist, const TensorList& bin_edges) {
    std::vector<int64_t> bin_ct(bins.size());
    std::transform(bins.begin(), bins.end(), bin_ct.begin(), [](Tensor t) { return t.numel() - 1; });
    histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
}

template<typename scalar_t>
void infer_bin_edges_from_input(const Tensor& input, const int64_t D,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    Tensor min, max;
    std::tie(min, max) = aminmax(input, 0);

    TORCH_INTERNAL_ASSERT(min.is_contiguous() && max.is_contiguous());


    TensorAccessor<scalar_t, 1> min_accessor = min.accessor<scalar_t, 1>();
    TensorAccessor<scalar_t, 1> max_accessor = max.accessor<scalar_t, 1>();

    for (int64_t dim = 0; dim < D; dim++) {
        leftmost_edges[dim] = min_accessor[dim];
        rightmost_edges[dim] = max_accessor[dim];
    }
}

/* Determines the outermost bin edges. For simplicity when calling into _aminmax,
 * assumes that input has already been reshaped to (N, D).
 */
std::pair<std::vector<double>, std::vector<double>>
select_outer_bin_edges(const Tensor& input, c10::optional<c10::ArrayRef<double>> range) {
    TORCH_INTERNAL_ASSERT(input.dim() == 2, "expected input to have shape (N, D)");
    const int64_t D = input.size(-1);

    // Default ranges for empty input matching numpy.histogram's default
    std::vector<double> leftmost_edges(D, 0.), rightmost_edges(D, 1.);

    if (range.has_value()) {
        // range is specified
        TORCH_CHECK(int64_t(range.value().size()) == 2 * D, "torch.histogramdd: for a ", D, "-dimensional histogram",
                " range should have ", 2 * D, " elements, but got ", range.value().size());

        for (int64_t dim = 0; dim < D; dim++) {
            leftmost_edges[dim] = range.value()[2 * dim];
            rightmost_edges[dim] = range.value()[2 * dim + 1];
        }
    } else if (input.numel() > 0) {
        // non-empty input
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogramdd", [&]() {
            infer_bin_edges_from_input<scalar_t>(input, D, leftmost_edges, rightmost_edges);
        });
    }

    for (int64_t dim = 0; dim < D; dim++) {
        double leftmost_edge = leftmost_edges[dim];
        double rightmost_edge = rightmost_edges[dim];

        TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
                std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
                "torch.histogramdd: dimension ", dim, "'s range [",
                leftmost_edge, ", ", rightmost_edge, "] is not finite");

        TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogramdd: min should not exceed max, but got",
                " min ", leftmost_edge, " max ", rightmost_edge, " for dimension ", dim);

        // Expand empty range to match numpy behavior and avoid division by 0 in normalization
        if (leftmost_edge == rightmost_edge) {
            leftmost_edges[dim] -= 0.5;
            rightmost_edges[dim] += 0.5;
        }
    }

    return std::make_pair(leftmost_edges, rightmost_edges);
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

/* Versions of histogramdd in which bins is a Tensor[] defining the sequences of bin edges.
 */
Tensor& histogramdd_out_cpu(const Tensor& self, TensorList bins,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    histogramdd_check_inputs(self, bins, weight);
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    // TODO: what if bin_edges doesn't have the correct length
    for (size_t dim = 0; dim < bins.size(); dim++) {
        bin_edges[dim].copy_(bins[dim]);
    }

    histogramdd_stub(self.device().type(), self, weight, density, hist, bin_edges);
    return hist;
}
Tensor histogramdd_cpu(const Tensor& self, TensorList bins,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    std::vector<Tensor> bin_edges_out;
    TensorList bin_edges_out_tl;
    histogramdd_out_cpu(self, bins, weight, density, hist, bin_edges_out_tl);
    return hist;
}
std::vector<Tensor> histogramdd_bin_edges_cpu(const Tensor& self, TensorList bins,
        const c10::optional<Tensor>& weight, bool density) {
    return { at::tensor(0) };
}

/* Versions of histogram in which bins is a Tensor defining the sequence of bin edges.
 */
std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, const Tensor& bins,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    c10::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bins;
    TensorList bins_out = bin_edges;

    histogramdd_out_cpu(reshaped_self, bins_in, reshaped_weight, density, hist, bins_out);

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
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    c10::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped_self, std::vector<int64_t>{bin_ct}, hist, bins_out);
    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);
    linspace_cpu_out(outer_bin_edges.first[0], outer_bin_edges.second[0], bin_ct + 1, bin_edges);

    histogramdd_check_inputs(reshaped_self, bins_in, reshaped_weight);

    histogram_linear_stub(reshaped_self.device().type(), reshaped_self, reshaped_weight, density, hist, bin_edges, true);
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

    Tensor reshaped = self.reshape({ self.numel(), 1 });
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped, std::vector<int64_t>{bin_ct}, hist, bins_out);

    auto outer_bin_edges = histc_select_outer_bin_edges(self, min, max);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1, bin_edges);

    histogramdd_check_inputs(reshaped, bins_in, {});

    histogram_linear_stub(reshaped.device().type(), reshaped,
            c10::optional<Tensor>(), false, hist, bin_edges, false);
    return hist;
}
Tensor histogram_histc_cpu(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    return histogram_histc_cpu_out(self, bin_ct, min, max, hist);
}

}} // namespace at::native
