#include <ATen/AccumulateType.h>
#include <c10/core/Scalar.h>
#include <limits>



namespace at::native {

inline void arange_check_bounds(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step) {
  // use double precision for validation to avoid precision issues
  double dstart = start.to<double>();
  double dend = end.to<double>();
  double dstep = step.to<double>();

  TORCH_CHECK(dstep > 0 || dstep < 0, "step must be nonzero");
  TORCH_CHECK(
      std::isfinite(dstart) && std::isfinite(dend),
      "unsupported range: ",
      dstart,
      " -> ",
      dend);
  TORCH_CHECK(
      ((dstep > 0) && (dend >= dstart)) || ((dstep < 0) && (dend <= dstart)),
      "upper bound and lower bound inconsistent with step sign");
}

template <typename scalar_t>
int64_t compute_arange_size(const Scalar& start, const Scalar& end, const Scalar& step) {
  arange_check_bounds(start, end, step);

  // we use double precision for (start - end) / step
  // to compute size_d for consistency across devices.
  // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 scalar_t,
  // but double on cpu for the same,
  // and the effective output size starts differing on CPU vs GPU because of precision issues, which
  // we dont want.
  // the corner-case we do want to take into account is int64_t, which has higher precision than double
  double size_d;
  if constexpr (std::is_same_v<scalar_t, int64_t>) {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();
    int64_t sgn = (xstep > 0) - (xstep < 0);
    size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
  } else {
    size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                        / step.to<double>());
  }

  TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");

  return static_cast<int64_t>(size_d);
}

} // namespace at::native
