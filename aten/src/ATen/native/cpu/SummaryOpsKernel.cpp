#include <ATen/native/SummaryOps.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>

namespace at {
namespace native {
namespace {

void histc_cpu_kernel(TensorIteratorBase& iter, Scalar min, Scalar max, const Tensor& result) {
  AT_DISPATCH_ALL_TYPES(iter.input_dtype(), "histc_cpu", [&] {
    const auto min_v = min.to<scalar_t>();
    const auto max_v = max.to<scalar_t>();

    TORCH_CHECK(std::isfinite(min_v) && std::isfinite(max_v),
                "range of [", min_v, ", ", max_v, "] is not finite");

    auto hist = result.accessor<scalar_t, 1>();
    const auto num_bins = hist.size(0);

    // calculate bin offset in at least float precision
    using div_t = decltype(min_v + 1.0f);
    const auto scale = num_bins / static_cast<div_t>(max_v - min_v);

    // Serial otherwise threads share a read-write dependency on histogram bins
    cpu_serial_kernel(iter, [&](scalar_t val) {
      if (val >= min_v && val <= max_v) {
        const auto bin = std::min(
            num_bins - 1, static_cast<int64_t>((val - min_v) * scale));
        hist[bin] += 1;
      }
    });
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(histc_stub, &histc_cpu_kernel);
}}  // namespace at::native
