#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Distributions.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/poisson_native.h>
#endif

namespace at { namespace native {

Tensor _s_poisson_cuda(const Tensor& lambda, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  launch_poisson_cuda_kernel(ret, lambda, gen);
  return ret;
}

Tensor _s_binomial_cuda(const Tensor& count, const Tensor& prob, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(count)
      .add_input(prob)
      .build();
  launch_binomial_cuda_kernel(iter, gen);
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  return ret;
}

Tensor _s_dirichlet_cuda(const Tensor& alpha, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(ret)
      .add_input(gamma_sum)
      .build();
  launch_dirichlet_kernel(iter);
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(self)
      .add_input(output)
      .build();
  launch_standard_gamma_grad_kernel(iter);
  return ret;
}

Tensor _dirichlet_grad_cuda(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(x)
      .add_input(alpha)
      .add_input(total)
      .build();
  launch_dirichlet_grad_kernel(iter);
  return ret;
}

}} // namespace at::native
