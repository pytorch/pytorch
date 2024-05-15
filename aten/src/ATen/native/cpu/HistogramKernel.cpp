#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Histogram.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <algorithm>
#include <numeric>
#include <functional>

namespace at::native {

namespace {

constexpr int64_t HISTOGRAM_GRAIN_SIZE = 200;

/* The main algorithm. Expects that the input tensor has shape (N, D).
 * Expects that bin_edges contains D one-dimensional tensors, each specifying
 * an increasing sequences of bin edges.
 *
 * Interprets the input as N different D-dimensional coordinates and maps them
 * into the D-dimensional bins defined by bin_edges, accumulating a D-dimensional
 * histogram in the hist tensor.
 *
 * Accepts a template argument of type BIN_SELECTION_ALGORITHM specifying how
 * the scalars in each dimension should be mapped into the dimension's bins:
 *
 *     - LINEAR_INTERPOLATION: each bin edge sequence must form a linear progression.
 *       Scalars are mapped to bins by computing
 *           (element - leftmost_edge)/(rightmost_edge - leftmost_edge) * bin_ct
 *       and truncating the result to an integer.
 *
 *       This is the fastest option, but its results may not be perfectly consistent
 *       with the boundaries specified in bin_edges due to precision issues.
 *
 *       Used by torch.histc, which doesn't need consistency with bin_edges as it does not
 *       return bin_edges. Additionally, this implementation is identical to the legacy histc
 *       implementation, which was replaced when histogram was implemented.
 *
 *     - LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH: Also expects that each bin edge sequence
 *       forms a linear progression. For each scalar, if 'pos' is the bin selected by the
 *       LINEAR_INTERPOLATION approach, this approach inspects the boundaries in bin_edges to
 *       place the scalar into pos - 1, pos, or pos + 1. The "local search" over neighboring
 *       bins allows for correction of misclassifications due to precision issues (a scalar
 *       very close to a bin_edge may be misclassified by LINEAR_INTERPOLATION).
 *
 *       Should produce the same output as the general case BINARY_SEARCH, but run about
 *       3x faster asymptotically.
 *
 *       Used by torch.histogram for cases in which bin_edges is constructed using
 *       torch.linspace. The behavior of LINEAR_INTERPOLATION may not perfectly align
 *       with linspace bin_edges due to precision issues. torch.histogram returns both
 *       the hist and bin_edges tensors as output, so the "local search" is needed to
 *       keep its output internally consistent.
 *
 *     - BINARY_SEARCH: Handles torch.histogram's general case by by searching over the
 *       elements of bin_edges. Implemented using std::upper_bound.
 *
 * See discussion at https://github.com/pytorch/pytorch/pull/58780#discussion_r648604866
 * for further details on relative performance of the bin selection algorithms.
 */
enum BIN_SELECTION_ALGORITHM {
    LINEAR_INTERPOLATION,
    LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
    BINARY_SEARCH,
};
template<typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_cpu_contiguous(Tensor& hist, const TensorList& bin_edges,
        const Tensor& input, const std::optional<Tensor>& weight) {
    TORCH_INTERNAL_ASSERT(input.dim() == 2);

    const int64_t N = input.size(0);
    if (weight.has_value()) {
        TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
    }

    const int64_t D = input.size(1);
    TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
    for (const auto dim : c10::irange(D)) {
        TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
        TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
    }

    if (D == 0) {
        // hist is an empty tensor in this case; nothing to do here
        return;
    }

    TensorAccessor<const input_t, 2> accessor_in = input.accessor<const input_t, 2>();

    /* Constructs a std::optional<TensorAccessor> containing an accessor if
     * the optional weight tensor has a value.
     */
    const auto accessor_wt = weight.has_value()
            ? std::optional<TensorAccessor<const input_t, 1>>(weight.value().accessor<const input_t, 1>())
            : std::optional<TensorAccessor<const input_t, 1>>();

    std::vector<input_t*> bin_seq(D);
    std::vector<int64_t> num_bin_edges(D);
    std::vector<input_t> leftmost_edge(D), rightmost_edge(D);

    for (const auto dim : c10::irange(D)) {
        bin_seq[dim] = bin_edges[dim].data_ptr<input_t>();
        num_bin_edges[dim] = bin_edges[dim].numel();
        leftmost_edge[dim] = bin_seq[dim][0];
        rightmost_edge[dim] = bin_seq[dim][num_bin_edges[dim] - 1];
    }

    int64_t GRAIN_SIZE = std::max(int64_t(1), HISTOGRAM_GRAIN_SIZE / D);

    /* Parallelizes processing of input using at::parallel_for.
     * Each thread accumulates a local result into their own slice of
     * thread_histograms which get summed together at the end.
     */
    const auto num_threads = at::get_num_threads();
    const auto hist_sizes = hist.sizes();
    DimVector thread_hist_sizes(hist_sizes.size() + 1);
    thread_hist_sizes[0] = num_threads;
    std::copy(hist_sizes.begin(), hist_sizes.end(),
              thread_hist_sizes.begin() + 1);
    Tensor thread_histograms = at::zeros(thread_hist_sizes, hist.dtype());
    TORCH_INTERNAL_ASSERT(thread_histograms.is_contiguous());

    at::parallel_for(0, N, GRAIN_SIZE, [&](int64_t start, int64_t end) {
        const auto tid = at::get_thread_num();
        auto hist_strides = thread_histograms.strides();
        input_t *hist_local_data = thread_histograms.data_ptr<input_t>();

        // View only this thread's local results
        hist_local_data += hist_strides[0] * tid;
        hist_strides = hist_strides.slice(1);

        for (const auto i : c10::irange(start, end)) {
            bool skip_elt = false;
            int64_t hist_index = 0;

            for (const auto dim : c10::irange(D)) {
                const input_t elt = accessor_in[i][dim];

                // Skips elements which fall outside the specified bins and NaN elements
                if (!(elt >= leftmost_edge[dim] && elt <= rightmost_edge[dim])) {
                    skip_elt = true;
                    break;
                }

                int64_t pos = -1;

                if (algorithm == BINARY_SEARCH) {
                    // Handles the general case via binary search on the bin edges.
                    pos = std::upper_bound(bin_seq[dim], bin_seq[dim] + num_bin_edges[dim], elt)
                            - bin_seq[dim] - 1;
                } else if (algorithm == LINEAR_INTERPOLATION
                        || algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                    /* When bin_edges is known to be a linear progression, maps elt to
                     * the appropriate bin via simple division.
                     */
                    pos = static_cast<int64_t>((elt - leftmost_edge[dim])
                            * (num_bin_edges[dim] - 1)
                            / (rightmost_edge[dim] - leftmost_edge[dim]));

                    /* Ensures consistency with bin_edges by checking the bins to the left and right
                     * of the selected position. Necessary for cases in which an element very close
                     * to a bin edge may be misclassified by simple division.
                     */
                    if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                        int64_t pos_min = std::max(static_cast<int64_t>(0), pos - 1);
                        int64_t pos_max = std::min(pos + 2, num_bin_edges[dim]);
                        pos = std::upper_bound(bin_seq[dim] + pos_min, bin_seq[dim] + pos_max, elt)
                                - bin_seq[dim] - 1;
                    }
                } else {
                    TORCH_INTERNAL_ASSERT(false);
                }

                // Unlike other bins, the rightmost bin includes its right boundary
                if (pos == (num_bin_edges[dim] - 1)) {
                    pos -= 1;
                }

                hist_index += hist_strides[dim] * pos;
            }

            if (!skip_elt) {
                // In the unweighted case, the default weight is 1
                input_t wt = accessor_wt.has_value() ? accessor_wt.value()[i] : static_cast<input_t>(1);

                hist_local_data[hist_index] += wt;
            }
        }
    });

    at::sum_out(hist, thread_histograms, /*dim=*/{0});
}

/* Some pre- and post- processing steps for the main algorithm.
 * Initializes hist to 0, calls into the main algorithm, and normalizes output if necessary.
 */
template<BIN_SELECTION_ALGORITHM bin_algorithm>
void histogramdd_out_cpu_template(const Tensor& self, const std::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
    hist.fill_(0);

    const int64_t N = self.size(-1);
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            (int64_t)1, std::multiplies<int64_t>());

    const Tensor reshaped_input = self.reshape({M, N});

    const auto reshaped_weight = weight.has_value()
            ? std::optional<Tensor>(weight.value().reshape({M}))
            : std::optional<Tensor>();

    std::vector<Tensor> bin_edges_contig(bin_edges.size());
    for (const auto dim : c10::irange(bin_edges_contig.size())) {
        bin_edges_contig[dim] = bin_edges[dim].contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "histogram_cpu", [&]() {
        histogramdd_cpu_contiguous<scalar_t, bin_algorithm>(
                hist, bin_edges_contig, reshaped_input, reshaped_weight);
    });

    /* Divides each bin's value by the total count/weight in all bins,
     * and by the bin's volume.
     */
    if (density) {
        const auto hist_sum = hist.sum().item();
        hist.div_(hist_sum);

         /* For each dimension, divides each bin's value
          * by the bin's length in that dimension.
          */
        for (const auto dim : c10::irange(N)) {
            const auto bin_lengths = bin_edges[dim].diff();

            // Used to reshape bin_lengths to align with the corresponding dimension of hist.
            std::vector<int64_t> shape(N, 1);
            shape[dim] = bin_lengths.numel();

            hist.div_(bin_lengths.reshape(shape));
        }
    }
}

/* The general implementation of the histogram kernel. Maps each element of the input tensor
 * to its corresponding bin by performing a binary search over the elements of bin_edges.
 *
 * Refer to histogramdd_out_cpu_template for more details.
 */
static void histogramdd_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
    histogramdd_out_cpu_template<BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

/* A faster version of the histogram kernel for cases in which bin_edges are known
 * to form a linear progression.
 *
 * Refer to histogramdd_out_cpu_template for more details.
 */
static void histogramdd_linear_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight,
        bool density, Tensor& hist, const TensorList& bin_edges, bool local_search) {
    if (local_search) {
        // histogramdd codepath: both hist and bin_edges are eventually returned as output,
        // so we'll keep them consistent
        histogramdd_out_cpu_template<LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
              self, weight, density, hist, bin_edges);
    } else {
        // histc codepath: bin_edges are not returned to the caller
        histogramdd_out_cpu_template<LINEAR_INTERPOLATION>(
              self, weight, density, hist, bin_edges);
    }
}

template<typename scalar_t>
void infer_bin_edges_from_input(const Tensor& input, const int64_t N,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    // Calls aminmax on input with dim=0, reducing all but the innermost dimension of input.
    Tensor min, max;
    std::tie(min, max) = aminmax(input, 0);

    TORCH_INTERNAL_ASSERT(min.is_contiguous() && max.is_contiguous());

    const scalar_t *min_data = min.const_data_ptr<scalar_t>();
    std::copy(min_data, min_data + N, leftmost_edges.begin());

    const scalar_t *max_data = max.const_data_ptr<scalar_t>();
    std::copy(max_data, max_data + N, rightmost_edges.begin());
}

static void histogram_select_outer_bin_edges_impl(const Tensor& input, const int64_t N,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogramdd", [&]() {
        infer_bin_edges_from_input<scalar_t>(input, N, leftmost_edges, rightmost_edges);
    });
}

} // namespace

REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel_impl);
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel_impl);
REGISTER_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_impl);

} // namespace at::native
