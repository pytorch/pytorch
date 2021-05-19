#include <ATen/native/Histogram.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BucketizationUtils.h>

#include <algorithm>
#include <mutex>
#include <tuple>

namespace at { namespace native {

namespace {

const int64_t HISTOGRAM_GRAIN_SIZE = 200;

template<typename input_t, typename output_t, typename ExtractWeight_t, bool LinearBinEdges>
void histogram_cpu_contiguous(Tensor& hist, const Tensor& bin_edges, const Tensor& input, const Scalar& min, const Scalar& max, ExtractWeight_t extract_weight) {
    int64_t numel_in = input.numel();
    if (!numel_in) {
        return;
    }

    int64_t numel_be = bin_edges.numel();
    TORCH_CHECK(numel_be > 0, "expected bin_edges to contain at least 1 element but it contains 0");

    const input_t *data_in = input.data_ptr<input_t>();
    const input_t *data_be = bin_edges.data_ptr<input_t>();

    output_t *data_out = hist.data_ptr<output_t>();

    input_t min_bound = min.to<input_t>();
    input_t max_bound = max.to<input_t>();

    if (min_bound == max_bound) {
        min_bound = input.min().item<input_t>();
        max_bound = input.max().item<input_t>();
    }
    if (min_bound == max_bound) {
        min_bound -= 1;
        max_bound += 1;
    }

    const input_t leftmost_bin_edge = data_be[0];
    const input_t rightmost_bin_edge = data_be[numel_be - 1];

    // we'll process only those elements which fall within the caller-specified [min, max]
    // range and also fall within the range of the specified bins
    min_bound = std::max(min_bound, leftmost_bin_edge);
    max_bound = std::min(max_bound, rightmost_bin_edge);

    std::mutex data_out_mutex;
    at::parallel_for(0, numel_in, HISTOGRAM_GRAIN_SIZE, [&](int64_t start, int64_t end) {
        std::vector<output_t> data_out_local(numel_be - 1, output_t(0));

        for (int64_t i = start; i < end; ++i) {
            if (min_bound <= data_in[i] && data_in[i] <= max_bound) {
                int64_t pos = -1;
                if (LinearBinEdges) {
                    pos = (int64_t)((data_in[i] - leftmost_bin_edge)
                            / (rightmost_bin_edge - leftmost_bin_edge)
                            * (numel_be - 1));
                } else {
                    pos = std::upper_bound(data_be, data_be + numel_be, data_in[i]) - data_be - 1;
                }

                // Unlike other bins, the rightmost bin includes its right boundary
                if (pos == (numel_be - 1))
                    --pos;

                data_out_local[pos] += extract_weight(i);
            }
        }

        const std::lock_guard<std::mutex> lock(data_out_mutex);
        for (int64_t i = 0; i < numel_be - 1; i++) {
            data_out[i] += data_out_local[i];
        }
    });
}

template<bool LinearBinEdges>
void dispatch(Tensor& hist, const Tensor& bin_edges, const Tensor& input, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density) {
    if (weight.has_value()) {
        ScalarType weight_dtype = weight.value().scalar_type();
        if (weight_dtype == ScalarType::Float) {
            const float *data_wt = weight.value().data_ptr<float>();
            auto extract_weight_float = [&data_wt](int64_t i) -> float {
                return data_wt[i];
            };
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
                histogram_cpu_contiguous<scalar_t, double, decltype(extract_weight_float), LinearBinEdges>
                    (hist, bin_edges, input, min, max, extract_weight_float);
            });
        } else if (weight_dtype == ScalarType::Double) {
            const double *data_wt = weight.value().data_ptr<double>();
            auto extract_weight_double = [&data_wt](int64_t i) -> double {
                return data_wt[i];
            };
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
                histogram_cpu_contiguous<scalar_t, double, decltype(extract_weight_double), LinearBinEdges>
                    (hist, bin_edges, input, min, max, extract_weight_double);
            });
        } else assert(false);

        return;
    }

    if (density) {
        auto extract_weight_double = [](int64_t i) -> double {
            return 1.0;
        };
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
            histogram_cpu_contiguous<scalar_t, double, decltype(extract_weight_double), LinearBinEdges>
                (hist, bin_edges, input, min, max, extract_weight_double);
        });
    } else {
        auto extract_weight_long = [](int64_t i) -> int64_t {
            return 1L;
        };
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
            histogram_cpu_contiguous<scalar_t, int64_t, decltype(extract_weight_long), LinearBinEdges>
                (hist, bin_edges, input, min, max, extract_weight_long);
        });
    }
}

inline void histogram_maybe_trim_input_tensors(
    Tensor& trimmed_input,
    Tensor& trimmed_bin_edges,
    Tensor& trimmed_weight,
    const Tensor& raw_input,
    const Tensor& raw_bin_edges,
    const c10::optional<Tensor>& weight) {
  // Reuse function from BucketizationUtils handling input and bin_edges
  searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_bin_edges, raw_input, raw_bin_edges);

  if (weight.has_value() && !weight.value().is_contiguous()) {
    TORCH_WARN_ONCE("input weight tensor is non-contiguous, this will lower the performance due to extra data copy "
        "when converting non-contiguous tensor to contiguous, please use contiguous input value tensor if possible");
    trimmed_weight = weight.value().contiguous();
  }
}

template<bool LinearBinEdges>
std::tuple<Tensor&, Tensor&>
histogram_out_cpu_template(const Tensor& self, const Tensor& bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    TORCH_CHECK(bins.numel() > 0, "bins tensor should have at least 1 element, but has 0");

    if (hist.numel() == 0) {
        hist.resize_({bins.numel() - 1});
    }

    if (bin_edges.numel() == 0) {
        bin_edges.resize_({bins.numel()});
    }

    TORCH_CHECK(bins.numel() == hist.numel() + 1, "bins tensor should have 1 more element than hist tensor, but",
            " we got bins(", bins.numel(), ") and hist(", hist.numel(), ")");

    TORCH_CHECK(bins.numel() == bin_edges.numel(), "bins tensor and bin_edges should have the same number of elements,",
            " but we got bins(", bins.numel(), ") and bin_edges(", bin_edges.numel(), ")");

    Tensor trimmed_input, trimmed_bins, trimmed_weight;
    histogram_maybe_trim_input_tensors(trimmed_input, trimmed_bins, trimmed_weight, self, bins, weight);

    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_bins = trimmed_bins.defined() ? trimmed_bins : bins;

    auto trimmed_weight_optional = c10::optional<Tensor>(trimmed_weight);
    const c10::optional<Tensor>& final_weight = trimmed_weight.defined() ? trimmed_weight_optional : weight;

    hist.fill_(0);

    dispatch<LinearBinEdges>(hist, final_bins, final_input, min, max, final_weight, density);

    if (density) {
        auto bin_widths = bins.diff().to(ScalarType::Double);
        auto hist_sum = hist.sum().item<double>();
        hist.div_(bin_widths);
        hist.div_(hist_sum);
    }

    bin_edges.copy_(bins);

    return {hist, bin_edges};
}

//static std::tuple<Tensor&, Tensor&>
static void
histogram_kernel_impl(const Tensor& self, const Tensor& bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
  histogram_out_cpu_template<false>(self, bins, min, max, weight, density, hist, bin_edges);
}

//static std::tuple<Tensor&, Tensor&>
static void
histogram_linear_kernel_impl(const Tensor& self, const Tensor& bins, const Scalar& min, const Scalar& max, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
  histogram_out_cpu_template<true>(self, bins, min, max, weight, density, hist, bin_edges);
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_stub, &histogram_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(histogram_linear_stub, &histogram_linear_kernel_impl);

}} // namespace at::native
