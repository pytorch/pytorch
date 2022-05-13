//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <cmath>
#include <limits>

namespace at {
namespace native {


Tensor& arange_mps_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_MPS_TYPES(result.scalar_type(), "arange_mps", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    double size_d;
    if (std::is_same<scalar_t, int64_t>::value) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                          / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                          / step.to<double>());
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
              std::isfinite(static_cast<double>(xend)),
              "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
              "upper bound and larger bound inconsistent with step sign");

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
              "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);
    int64_t numel = result.numel();

    if (numel != size) {
      if(numel > 0){
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

    //TODO: Add arange Metal kernel.

    if(!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}
}} // namespace at::native
