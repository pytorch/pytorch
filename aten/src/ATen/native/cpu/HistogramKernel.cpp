#include <ATen/native/Histogram.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <mutex>
#include <tuple>

namespace at { namespace native {

namespace {

constexpr int64_t HISTOGRAM_GRAIN_SIZE = 200;

/* The main algorithm. Maps the elements of input into the bins defined by bin_edges.
 * Expects that the elements of bin_edges are increasing; behavior is otherwise undefined.
 *
 * Accepts a template argument of type BIN_SELECTION_ALGORITHM specifying how the
 * elements of input should be mapped into the bins:
 *
 *     - LINEAR_INTERPOLATION: bin_edges must contain a linear progression.
 *       Elements of input are mapped to bins by computing
 *           (element - leftmost_edge)/(rightmost_edge - leftmost_edge) * bin_ct
 *       and truncating the result to an integer.
 *
 *       Results may not be perfectly consistent with the boundaries specified in bin_edges
 *       due to precision issues.
 *
 *       Used by torch.histc, which doesn't need consistency with bin_edges as it does not
 *       return bin_edges. Additionally, this implementation is identical to the legacy histc
 *       implementation, which was replaced when histogram was implemented.
 *
 *     - LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH: Also expects that bin_edges contains a
 *       linear progression. For each element, if 'pos' is the bin selected by the
 *       LINEAR_INTERPOLATION approach, this approach inspects the boundaries in bin_edges to
 *       place the element into pos - 1, pos, or pos + 1. The "local search" over neighboring
 *       bins allows for correction of misclassifications due to precision issues (an element
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
 *
 * Accumulates the total weight in each bin into the hist tensor.
 */
enum BIN_SELECTION_ALGORITHM {
    LINEAR_INTERPOLATION,
    LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
    BINARY_SEARCH,
};
template<typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogram_cpu_contiguous(Tensor& hist, const Tensor& bin_edges,
        const Tensor& input, const c10::optional<Tensor>& weight) {
    TORCH_INTERNAL_ASSERT(hist.is_contiguous());
    TORCH_INTERNAL_ASSERT(bin_edges.is_contiguous());
    TORCH_INTERNAL_ASSERT(hist.numel() + 1 == bin_edges.numel());
    TORCH_INTERNAL_ASSERT(input.dim() == 1);
    TORCH_INTERNAL_ASSERT(!weight.has_value() || weight.value().dim() == 1);

    const int64_t numel_in = input.numel();

    TensorAccessor<input_t, 1> accessor_in = input.accessor<input_t, 1>();

    /* Constructs a c10::optional<TensorAccessor> containing an accessor iff
     * the optional weight tensor has a value.
     */
    const auto accessor_wt = weight.has_value()
            ? c10::optional<TensorAccessor<input_t, 1>>(weight.value().accessor<input_t, 1>())
            : c10::optional<TensorAccessor<input_t, 1>>();

    const int64_t numel_be = bin_edges.numel();
    const input_t *data_be = bin_edges.data_ptr<input_t>();

    const input_t leftmost_bin_edge = data_be[0];
    const input_t rightmost_bin_edge = data_be[numel_be - 1];

    input_t *data_out = hist.data_ptr<input_t>();

    /* Parallelizes processing of input using at::parallel_for.
     * Each thread accumulates a local result for some range of the input in data_out_local
     * before locking data_out_mutex and adding its accumulated results to data_out.
     */
    std::mutex data_out_mutex;
    at::parallel_for(0, numel_in, HISTOGRAM_GRAIN_SIZE, [&](int64_t start, int64_t end) {
        // Allocates a buffer for the thread's local results
        std::vector<input_t> data_out_local(numel_be - 1, input_t(0));

        for (int64_t i = start; i < end; ++i) {
            const input_t elt = accessor_in[i];

            // Skips elements which fall outside the specified bins
            if (elt < leftmost_bin_edge || rightmost_bin_edge < elt) {
                continue;
            }

            int64_t pos = -1;

            if (algorithm == BINARY_SEARCH) {
                // Handles the general case via binary search on the bin edges.
                pos = std::upper_bound(data_be, data_be + numel_be, elt) - data_be - 1;
            } else if (algorithm == LINEAR_INTERPOLATION
                    || algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                /* When bin_edges is known to be a linear progression, maps elt to
                 * the appropriate bin via simple division.
                 */
                pos = static_cast<int64_t>((elt - leftmost_bin_edge)
                        / (rightmost_bin_edge - leftmost_bin_edge)
                        * (numel_be - 1));

                /* Ensures consistency with bin_edges by checking the bins to the left and right
                 * of the selected position. Necessary for cases in which an element very close
                 * to a bin edge may be misclassified by simple division.
                 */
                if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                    int64_t pos_min = std::max(static_cast<int64_t>(0), pos - 1);
                    int64_t pos_max = std::min(pos + 2, numel_be);
                    pos = std::upper_bound(data_be + pos_min, data_be + pos_max, elt) - data_be - 1;
                }
            } else {
                TORCH_INTERNAL_ASSERT(false);
            }

            // Unlike other bins, the rightmost bin includes its right boundary
            if (pos == (numel_be - 1)) {
                pos -= 1;
            }

            // In the unweighted case, the default weight is 1
            input_t wt = accessor_wt.has_value() ? accessor_wt.value()[i] : static_cast<input_t>(1);
            data_out_local[pos] += wt;
        }

        // Locks and updates the common output
        const std::lock_guard<std::mutex> lock(data_out_mutex);
        for (int64_t i = 0; i < numel_be - 1; i++) {
            data_out[i] += data_out_local[i];
        }
    });
}

/* Some pre- and post- processing steps for the main algorithm.
 * Initializes hist to 0, calls into the main algorithm, and normalizes output if necessary.
 */
template<BIN_SELECTION_ALGORITHM bin_algorithm>
void histogram_out_cpu_template(const Tensor& self, const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, const Tensor& bin_edges) {
    hist.fill_(0);

    const int64_t numel_in = self.numel();
    const Tensor reshaped_input = self.reshape({numel_in});

    const auto reshaped_weight = weight.has_value()
            ? c10::optional<Tensor>(weight.value().reshape({numel_in}))
            : c10::optional<Tensor>();

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histogram_cpu", [&]() {
        histogram_cpu_contiguous<scalar_t, bin_algorithm>(
                hist, bin_edges.contiguous(), reshaped_input, reshaped_weight);
    });

    // Converts the bin totals to a probability density function
    if (density) {
        auto bin_widths = bin_edges.diff();
        auto hist_sum = hist.sum().item();
        hist.div_(bin_widths);
        hist.div_(hist_sum);
    }
}

/* The general implementation of the histogram kernel. Maps each element of the input tensor
 * to its corresponding bin by performing a binary search over the elements of bin_edges.
 *
 * Refer to histogram_out_cpu_template for more details.
 */
static void histogram_kernel_impl(const Tensor& self, const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, const Tensor& bin_edges) {
    histogram_out_cpu_template<BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

/* A faster version of the histogram kernel for cases in which bin_edges are known
 * to form a linear progression.
 *
 * Refer to histogram_out_cpu_template for more details.
 */
static void histogram_linear_kernel_impl(const Tensor& self, const c10::optional<Tensor>& weight,
        bool density, Tensor& hist, const Tensor& bin_edges, bool local_search) {
    if (local_search) {
        // histogram codepath: both hist and bin_edges are eventually returned as output,
        // so we'll keep them consistent
        histogram_out_cpu_template<LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
              self, weight, density, hist, bin_edges);
    } else {
        // histc codepath: bin_edges are not returned to the caller
        histogram_out_cpu_template<LINEAR_INTERPOLATION>(
              self, weight, density, hist, bin_edges);
    }
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_stub, &histogram_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_linear_stub, &histogram_linear_kernel_impl);

}} // namespace at::native
