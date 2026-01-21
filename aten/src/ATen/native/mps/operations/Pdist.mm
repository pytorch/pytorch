#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Distance.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_forward.h>
#include <ATen/ops/_pdist_forward_native.h>
#include <ATen/ops/triu_indices.h>
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

} // namespace mps

void pdist_forward_kernel_mps(
    Tensor& result,
    const Tensor& self,
    const double p) {
  mps::pdist_forward_kernel_mps_impl(result, self, p);
}

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_mps)

} // namespace at::native
