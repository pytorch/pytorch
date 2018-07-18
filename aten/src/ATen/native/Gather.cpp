// Returns unique elements of input tensor.

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"

#include <tuple>

namespace at {
namespace native{

namespace {

std::tuple<std::vector<int64_t>, std::vector<int64_t> > gather_infer_size(IntList tensor, IntList indices, long dim) {
  auto dimsA = tensor.size();
  auto dimsB = indices.size();
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizesB(ndim);
  int64_t expandedSizeA;

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? tensor[dimA] : 1;
    long sizeB = (dimB >= 0) ? indices[dimB] : 1;

    if (i == dim) {
      expandedSizeA = sizeA;
      expandedSizesB[i] = sizeB;
    } else {
      AT_CHECK(
          sizeA == sizeB || sizeA == 1 || sizeB == 1,
          "The size of tensor a (", sizeA,
          ") must match the size of tensor b (", sizeB,
          ") at non-singleton dimension ", i);

      // 1s map to the other size (even 0).
      expandedSizesB[i] = sizeA == 1 ? sizeB : sizeA;
    }
  }

  std::vector<int64_t> expandedSizesA(expandedSizesB);
  expandedSizesA[dim] = expandedSizeA;
  return std::make_tuple(expandedSizesA, expandedSizesB);
}

inline std::tuple<Tensor, Tensor> gather_expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, long dim) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  std::vector<int64_t> expanded_size1, expanded_size2;
  std::tie(expanded_size1, expanded_size2) = gather_infer_size(to_expand1.sizes(), to_expand2.sizes(), dim);
  return std::make_tuple(
      to_expand1.expand(expanded_size1, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size2, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor> gather_expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name, long dim) {
  check_defined({to_expand1, to_expand2}, api_name);
  return gather_expand_outplace(to_expand1, to_expand2, dim);
}

} // namespace


Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor b_self, b_index;
  std::tie(b_self, b_index) = gather_expand_outplace(self, index, "gather", dim);
  return _s_gather_out(result, b_self, dim, b_index);
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor b_self, b_index;
  std::tie(b_self, b_index) = gather_expand_outplace(self, index, "gather", dim);
  return _s_gather(b_self, dim, b_index);
}

}  // namespace native
}  // namespace at
