#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Distance.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_backward.h>
#include <ATen/ops/_cdist_forward.h>
#include <ATen/ops/_pdist_backward_native.h>
#include <ATen/ops/_pdist_forward_native.h>
#include <ATen/ops/triu_indices.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace mps {

// pdist implementation for MPS
// We leverage the existing cdist implementation and extract the upper triangular part
static void pdist_forward_kernel_mps_impl(
    Tensor& result,
    const Tensor& self,
    const double p) {

  int64_t n = self.size(0);

  // Compute full distance matrix using cdist
  // cdist(self, self) gives us an n x n matrix of all pairwise distances
  Tensor full_distances = at::_cdist_forward(self, self, p, std::nullopt);

  // Extract upper triangular indices (excluding diagonal)
  // The result should be of size n*(n-1)/2
  auto indices = at::triu_indices(n, n, 1, self.options().dtype(kLong).device(self.device()));

  // Get row and column indices
  auto row_indices = indices[0];
  auto col_indices = indices[1];

  // Extract the upper triangular elements
  // full_distances[row_indices, col_indices]
  result.copy_(full_distances.index({row_indices, col_indices}));
}

static void pdist_backward_kernel_mps_impl(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& pdist) {

  int64_t n = self.size(0);

  if (n <= 1) {
    result.zero_();
    return;
  }

  // Create a full n x n gradient matrix from the upper triangular gradient
  Tensor full_grad = at::zeros({n, n}, grad.options());

  // Get upper triangular indices
  auto indices = at::triu_indices(n, n, 1, self.options().dtype(kLong).device(self.device()));
  auto row_indices = indices[0];
  auto col_indices = indices[1];

  // Fill upper triangular part
  full_grad.index_put_({row_indices, col_indices}, grad);

  // Make symmetric (lower triangular = upper triangular transpose)
  full_grad = full_grad + full_grad.transpose(0, 1);

  // Reconstruct full distance matrix
  Tensor full_distances = at::_cdist_forward(self, self, p, std::nullopt);

  // Use cdist backward to compute gradients
  // cdist_backward gives gradient w.r.t. x1 when x1 == x2
  Tensor grad_self = at::_cdist_backward(full_grad, self, self, p, full_distances);

  result.copy_(grad_self);
}

} // namespace mps

void pdist_forward_kernel_mps(
    Tensor& result,
    const Tensor& self,
    const double p) {
  mps::pdist_forward_kernel_mps_impl(result, self, p);
}

void pdist_backward_kernel_mps(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& pdist) {
  mps::pdist_backward_kernel_mps_impl(result, grad, self, p, pdist);
}

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_mps)
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_mps)

} // namespace at::native
