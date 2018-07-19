#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/WrapDimUtils.h"

#include <tuple>
#include <iostream>

namespace at {
namespace native{

namespace {

std::tuple<std::vector<int64_t>, std::vector<int64_t> > gather_infer_size(IntList tensor, IntList index, int64_t &dim) {
  auto dimsA = tensor.size();
  auto dimsB = index.size();
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;
  dim = maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> expandedSizesB(ndim);
  int64_t expandedSizeA = -1;

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? tensor[dimA] : 1;
    long sizeB = (dimB >= 0) ? index[dimB] : 1;

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

inline std::tuple<Tensor, Tensor> gather_expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, int64_t &dim) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  std::vector<int64_t> expanded_size1, expanded_size2;
  std::tie(expanded_size1, expanded_size2) = gather_infer_size(to_expand1.sizes(), to_expand2.sizes(), dim);
  return std::make_tuple(
      to_expand1.expand(expanded_size1, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size2, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor> gather_expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name, int64_t &dim) {
  check_defined({to_expand1, to_expand2}, api_name);
  return gather_expand_outplace(to_expand1, to_expand2, dim);
}

inline Tensor broadcast_scatter(Tensor &self, const Tensor & index, const char *api_name, int64_t &dim) {
  check_defined({self, index}, api_name);
  std::vector<int64_t> b_self_sizes, b_index_sizes;
  std::tie(b_self_sizes, b_index_sizes) = gather_infer_size(self.sizes(), index.sizes(), dim);
  AT_CHECK(self.sizes().equals(b_self_sizes), "Broadcasting of scatter_ should not change shape of self");
  return index.expand(b_index_sizes, /*implicit=*/true);
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

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  Tensor b_index = broadcast_scatter(self, index, "scatter_", dim);
  return self._s_scatter_(dim, b_index, src);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar src) {
  Tensor b_index = broadcast_scatter(self, index, "scatter_", dim);
  return self._s_scatter_(dim, b_index, src);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  Tensor b_index = broadcast_scatter(self, index, "scatter_add_", dim);
  return self._s_scatter_add_(dim, b_index, src);
}

}  // namespace native
}  // namespace at
