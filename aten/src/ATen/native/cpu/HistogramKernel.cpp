#include <ATen/native/Histogram.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <mutex>
#include <tuple>

namespace at { namespace native {

namespace {

const int64_t HISTOGRAM_GRAIN_SIZE = 200;

template<typename input_t, typename ExtractWeight_t, bool LinearBinEdges>
void histogram_cpu_contiguous(Tensor& hist, const Tensor& bin_edges, const Tensor& input, ExtractWeight_t extract_weight) {
    int64_t numel_in = input.numel();
    if (!numel_in) {
        return;
    }

    int64_t numel_be = bin_edges.numel();

    const input_t *data_in = input.data_ptr<input_t>();
    const input_t *data_be = bin_edges.data_ptr<input_t>();

    input_t *data_out = hist.data_ptr<input_t>();

    const input_t leftmost_bin_edge = data_be[0];
    const input_t rightmost_bin_edge = data_be[numel_be - 1];

    std::mutex data_out_mutex;
    at::parallel_for(0, numel_in, HISTOGRAM_GRAIN_SIZE, [&](int64_t start, int64_t end) {
        std::vector<input_t> data_out_local(numel_be - 1, input_t(0));

        for (int64_t i = start; i < end; ++i) {
            if (leftmost_bin_edge <= data_in[i] && data_in[i] <= rightmost_bin_edge) {
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
void dispatch(Tensor& hist, const Tensor& bin_edges, const Tensor& input, const c10::optional<Tensor>& weight, bool density) {
    if (weight.has_value()) {
        ScalarType weight_dtype = weight.value().scalar_type();
        if (weight_dtype == ScalarType::Float) {
            const float *data_wt = weight.value().data_ptr<float>();
            auto extract_weight_float = [&data_wt](int64_t i) -> float {
                return data_wt[i];
            };
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
                histogram_cpu_contiguous<scalar_t, decltype(extract_weight_float), LinearBinEdges>
                    (hist, bin_edges, input, extract_weight_float);
            });
        } else if (weight_dtype == ScalarType::Double) {
            const double *data_wt = weight.value().data_ptr<double>();
            auto extract_weight_double = [&data_wt](int64_t i) -> double {
                return data_wt[i];
            };
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
                histogram_cpu_contiguous<scalar_t, decltype(extract_weight_double), LinearBinEdges>
                    (hist, bin_edges, input, extract_weight_double);
            });
        } else assert(false);

        return;
    }

    if (density) {
        auto extract_weight_float = [](int64_t i) -> float {
            return 1.0;
        };
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
            histogram_cpu_contiguous<scalar_t, decltype(extract_weight_float), LinearBinEdges>
                (hist, bin_edges, input, extract_weight_float);
        });
    } else {
        auto extract_weight_long = [](int64_t i) -> int64_t {
            return 1L;
        };
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "histogram_out_cpu", [&] {
            histogram_cpu_contiguous<scalar_t, decltype(extract_weight_long), LinearBinEdges>
                (hist, bin_edges, input, extract_weight_long);
        });
    }
}

template<bool LinearBinEdges>
std::tuple<Tensor&, Tensor&>
histogram_out_cpu_template(const Tensor& self, const c10::optional<Tensor>& weight, bool density, Tensor& hist, Tensor& bin_edges) {
    hist.fill_(0);
    dispatch<LinearBinEdges>(hist, bin_edges, self, weight, density);

    if (density) {
        auto bin_widths = bin_edges.diff();
        auto hist_sum = hist.sum().item();
        hist.div_(bin_widths);
        hist.div_(hist_sum);
    }

    return {hist, bin_edges};
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
