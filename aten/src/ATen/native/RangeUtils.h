#include <ATen/AccumulateType.h>
#include <c10/core/Scalar.h>
#include <limits>



namespace at::native {

inline void arange_check_bounds(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step) {
  if(isComplexType(start.type()) || isComplexType(end.type()) || isComplexType(step.type())) {
    auto startc = start.to<c10::complex<double>>();
    auto endc = end.to<c10::complex<double>>();
    auto stepc = step.to<c10::complex<double>>();

    TORCH_CHECK(stepc.real() != 0 || stepc.imag() != 0, "step must be nonzero");
    TORCH_CHECK(((stepc.real() > 0) && (endc.real() >= startc.real())) ||
        ((stepc.real() <= 0) && (endc.real() <= startc.real())),
        "upper bound and lower bound inconsistent with step sign for real part");
    TORCH_CHECK(((stepc.imag() > 0) && (endc.imag() >= startc.imag())) ||
        ((stepc.imag() <= 0) && (endc.imag() <= startc.imag())),
        "upper bound and lower bound inconsistent with step sign for imaginary part");
    TORCH_CHECK(
        std::isfinite(startc.real()) && std::isfinite(endc.real()),
        "unsupported range for real part: ",
        startc.real(),
        " -> ",
        endc.real());
    TORCH_CHECK(
        std::isfinite(startc.imag()) && std::isfinite(endc.imag()),
        "unsupported range for real part: ",
        startc.imag(),
        " -> ",
        endc.imag());

    return;
  }

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
    if (start.isIntegral(false) && end.isIntegral(false) && step.isIntegral(false)) {
      using accscalar_t = at::acc_type<scalar_t, false>;
      auto xstart = start.to<accscalar_t>();
      auto xend = end.to<accscalar_t>();
      auto xstep = step.to<accscalar_t>();
      TORCH_CHECK_VALUE(xstep != 0, "step must be nonzero");
      int64_t sgn = (xstep > 0) - (xstep < 0);
      size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
    } else {
      size_d = std::ceil((end.to<double>() - start.to<double>())
                          / step.to<double>());
    }
  } else if (isComplexType(start.type()) || isComplexType(end.type()) || isComplexType(step.type())) {
    using step_t = std::conditional_t<std::is_same_v<scalar_t, c10::complex<double>>, c10::complex<double>, c10::complex<float>>;
    auto xstartc = start.to<step_t>();
    auto xendc = end.to<step_t>();
    auto xstepc = step.to<step_t>();
    auto distance = xendc - xstartc;

    TORCH_CHECK(!(xstepc.real() == 0 && xstepc.imag() == 0), "complex step must be nonzero");
    if(xstepc.real() == 0) {
      int64_t sgn = (xstepc.imag() > 0) - (xstepc.imag() < 0);
      size_d = std::ceil((distance.imag() + xstepc.imag() - sgn) / xstepc.imag());
    } else if (xstepc.imag() == 0) {
      int64_t sgn = (xstepc.real() > 0) - (xstepc.real() < 0);
      size_d = std::ceil((distance.real() + xstepc.real() - sgn) / xstepc.real());
    } else {
      int64_t sgn_real = (xstepc.real() > 0) - (xstepc.real() < 0);
      int64_t sgn_imag = (xstepc.imag() > 0) - (xstepc.imag() < 0);
      auto size_d_real = std::ceil((distance.real() + xstepc.real() - sgn_real) / xstepc.real());
      auto size_d_imag = std::ceil((distance.imag() + xstepc.imag() - sgn_imag) / xstepc.imag());
      TORCH_CHECK(size_d_real == size_d_imag,
                  "cannot perform step due to incorrect upper and lower bounds")
        size_d = size_d_real; // size_d_imag is expected to be same
    }
  } else {
    size_d = std::ceil((end.to<double>() - start.to<double>())
                        / step.to<double>());
  }

  TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");

  return static_cast<int64_t>(size_d);
}

} // namespace at::native
