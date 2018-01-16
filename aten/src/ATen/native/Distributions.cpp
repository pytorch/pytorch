#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor& bernoulli_(Tensor& self, const Tensor& p, Generator* generator) {
  self.copy_(at::bernoulli(std::get<0>(expand_inplace(self, p)), generator));
  return self;
}

Tensor& bernoulli_(Tensor& self, double p, Generator* generator) {
  Tensor probs = self.type().toScalarType(kDouble).tensor({}).fill_(p);
  return native::bernoulli_(self, probs, generator);
}


// TODO Replace this with more accurate digamma().
template <typename scalar>
static inline scalar digamma_one(scalar x) {
  const double eps = x * 1e-3;
  return (std::lgamma(x + eps) - std::lgamma(x - eps)) / (eps + eps);
}

// Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
// for random number x drawn from a standard Gamma distribution Gamma(alpha).
template <typename scalar>
static inline scalar standard_gamma_grad_one(scalar alpha, scalar x) {
  // Use a Taylor series expansion for small x.
  if (x < 0.8f) {
    scalar numer = 1;
    scalar denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / i;
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = std::pow(x, alpha);
    const auto gamma_pdf = std::pow(x, alpha - 1) * std::exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha = (std::log(x) - digamma_one(alpha)) * gamma_cdf
        - pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return std::isnan(result) ? 0 : result;
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return numer_1 * numer_2 / denom;
    }
    const auto denom = std::sqrt(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = std::pow(x - alpha - alpha * std::log(x / alpha), -1.5f);
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 = std::log(x / alpha) * term23
                     - std::sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return -stirling * numer / denom;
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = std::log(x / alpha);
  const auto v = std::log(alpha);
  static const scalar coef_uv[3][8] = {
    {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
     1, 0.32668115, 0.10406089, 0.0014179084},
    {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
     0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
    {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
     0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
  };
  scalar coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return std::exp(p / q);
}

template <typename scalar>
struct StandardGammaGradOp {
  static void apply(Tensor& ret, const Tensor& self, const Tensor& output) {
    CPU_tensor_apply3<scalar, scalar, scalar>(ret, self, output,
      [](scalar& ret_val, const scalar& self_val, const scalar &output_val) {
         ret_val = standard_gamma_grad_one(self_val, output_val);
      }
    );
  }
};

Tensor _standard_gamma_grad_cpu(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  dispatch_floating_types<void, StandardGammaGradOp>(self.type(), "_standard_gamma_grad", ret, self, output);
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  runtime_error("_standard_gamma_grad is not implemented for CUDA types");
}

}
}
