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
  const scalar eps = x * 1e-2;
  return (std::lgamma(x + eps) - std::lgamma(x - eps)) / (eps + eps);
}

/** Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
    for random number x drawn from a standard Gamma distribution Gamma(alpha).
*/
template <typename scalar>
static inline scalar standard_gamma_grad_one(scalar alpha, scalar x) {
  // Use an asymptotic approximation for small x.
  if (x < 0.2f) {
    const auto a0 = 1 / alpha;
    const auto a1 = 1 / (alpha + 1);
    const auto a2 = 1 / (alpha + 2);
    const auto pow_x_alpha = std::pow(x, alpha);
    const auto gamma_pdf = std::pow(x, alpha - 1) * std::exp(-x);
    const auto gamma_cdf = pow_x_alpha * (a0 - x*a1 + 0.5f*x*x*a2);
    const auto gamma_cdf_alpha = (std::log(x) - digamma_one(alpha)) * gamma_cdf
        - pow_x_alpha * (a0*a0 - x*a1*a1 + 0.5f*x*x*a2*a2);
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return std::isnan(result) ? 0 : result;
  }

  // Use an asymptotic approximation for large alpha.
  if (alpha > 50.0f) {
    return std::sqrt(x / alpha);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = std::log(x / alpha);
  const auto v = std::log(alpha);
  static const scalar coef_uv[3][8] = {
    {0.16028008, -0.088064309, 0.019630876, -0.0016920282,
     1.0, 0.36659853, 0.10843863, 0.0066895454},
    {0.521894, 0.16095838, 0.06237597, 0.0023884253,
     0.083457714, 0.0073297628, -0.0059299053, -0.00093720389},
    {-0.0031143957, -0.012143877, -0.0057656484, -0.00064847254,
     0.0087262576, -0.00022820524, 1.8871047e-05, 9.6307964e-06},
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
