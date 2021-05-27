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
 * Accumulates the total weight in each bin into the hist tensor.
 */
template<typename input_t, bool LinearBinEdges>
void histogram_cpu_contiguous(Tensor& hist, const Tensor& bin_edges,
        const Tensor& input, const c10::optional<Tensor>& weight) {
    TORCH_INTERNAL_ASSERT(hist.is_contiguous());
    TORCH_INTERNAL_ASSERT(bin_edges.is_contiguous());
    TORCH_INTERNAL_ASSERT(hist.numel() + 1 == bin_edges.numel());
    TORCH_INTERNAL_ASSERT(input.dim() == 1);
    TORCH_INTERNAL_ASSERT(!weight.has_value() || weight.value().dim() == 1);

    const int64_t numel_in = input.numel();
    if (!numel_in) {
        return;
    }

    TensorAccessor<input_t, 1> accessor_in = input.accessor<input_t, 1>();

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
            if (LinearBinEdges) {
                // When bin_edges is known to be a linear progression, maps data_in[i] to
                // the appropriate bin via simple division.
                pos = static_cast<int64_t>((elt - leftmost_bin_edge)
                        / (rightmost_bin_edge - leftmost_bin_edge)
                        * (numel_be - 1));
            } else {
                // Handles the general case via binary search on the bin edges.
                pos = std::upper_bound(data_be, data_be + numel_be, elt) - data_be - 1;
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
template<bool LinearBinEdges>
std::tuple<Tensor&, Tensor&>
histogram_out_cpu_template(const Tensor& self, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    hist.fill_(0);

    const int64_t numel_in = self.numel();
    const Tensor reshaped_input = self.reshape({numel_in});

    const auto reshaped_weight = weight.has_value()
            ? c10::optional<Tensor>(weight.value().reshape({numel_in}))
            : c10::optional<Tensor>();

    switch (self.scalar_type()) {
        case ScalarType::Double: {
            histogram_cpu_contiguous<double, LinearBinEdges>(hist, bin_edges.contiguous(), reshaped_input, reshaped_weight);
            break;
        }
        case ScalarType::Float: {
            histogram_cpu_contiguous<float, LinearBinEdges>(hist, bin_edges.contiguous(), reshaped_input, reshaped_weight);
            break;
        }
        default:
            TORCH_INTERNAL_ASSERT(false, "histogram_out not supported on CPUType for ", self.scalar_type());
    }

    if (density) {
        auto bin_widths = bin_edges.diff();
        auto hist_sum = hist.sum().item();
        hist.div_(bin_widths);
        hist.div_(hist_sum);
    }

    return std::forward_as_tuple(hist, bin_edges);
}

//static std::tuple<Tensor&, Tensor&>
static void
histogram_kernel_impl(const Tensor& self, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    histogram_out_cpu_template<false>(self, weight, density, hist, bin_edges);
}

//static std::tuple<Tensor&, Tensor&>
static void
histogram_linear_kernel_impl(const Tensor& self, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    histogram_out_cpu_template<true>(self, weight, density, hist, bin_edges);
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_stub, &histogram_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_linear_stub, &histogram_linear_kernel_impl);

}} // namespace at::native
