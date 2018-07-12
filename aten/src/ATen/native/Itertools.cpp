// Returns unique elements of input tensor.

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"

#include <vector>

namespace at {
namespace native{

std::vector<Tensor> cartesian_prod(const TensorList& tensors) {
  std::vector<Tensor> grids = at::meshgrid(tensors);
  for(Tensor &t : grids) {
    t = t.flatten();
  }
  return grids;
}

std::vector<Tensor> combinations(const Tensor& self, int64_t r, bool with_replacement) {
  AT_CHECK(self.dim() == 1, "Expect a 1D vector, but got", tensor);
  AT_CHECK(r > 0, "Expect a positive number, but got", r);
  int64_t num_elements = self.numel();
  auto grids = at::meshgrid(std::vector<Tensor>(r, self));

  // get a mask that has value 1 whose indices satisfies i + j + k + ... <= n
  // or i + j + k + ... < n (depending on with_replacement)
  auto range = at::arange(self.type().toScalarType(kLong), num_elements);
  auto index_grids = at::meshgrid(std::vector<Tensor>(r, range));
  auto sum_index_grids = index_grids[0];
  std::for_each(index_grids.begin() + 1, index_grids.end(), [&](const Tensor &t){
    sum_index_grids += t;
  });
  auto mask = with_replacement ? sum_index_grids <= num_elements : sum_index_grids < num_elements;

  for(Tensor &t : grids) {
    t = t[mask];
  }
  return grids;
}

}  // namespace native
}  // namespace at
