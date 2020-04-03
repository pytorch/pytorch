#include <ATen/native/SummaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>


namespace at { namespace native { namespace {

void histc_kernel(
    TensorIterator& iter,
    Tensor& hist,
    int64_t nbins,
    Scalar minvalue,
    Scalar maxvalue
) {
  const Tensor& tensor = iter.tensor(0);
  AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), "histc_cpu", [&]() -> void {
    scalar_t minval;
    scalar_t maxval;
    int64_t *h_data;

    minval = minvalue.to<scalar_t>();
    maxval = maxvalue.to<scalar_t>();

    if (minval == maxval) {
      minval = tensor.min().item().to<scalar_t>();
      maxval = tensor.max().item().to<scalar_t>();
    }
    if (minval == maxval) {
      minval = minval - 1;
      maxval = maxval + 1;
    }

    TORCH_CHECK(!(std::isinf(minval) || std::isinf(maxval) || std::isnan(minval) || std::isnan(maxval)), "range of [", minval, ", ", maxval, "] is not finite");
    TORCH_CHECK(minval < maxval, "max must be larger than min");

    h_data = hist.data_ptr<int64_t>();
    cpu_serial_kernel(iter, [&](scalar_t val) -> void { 
      if (val >= minval && val <= maxval) {
        const int64_t bin = (int64_t)((val-minval) / (maxval-minval) * nbins);
        h_data[std::min(bin, nbins-1)] += 1;
      }
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(histc_stub, &histc_kernel);

}} // namespace at::native
