/*

The broadcasting of `gather` works according to steps described below:
Let's say we are doing a `t.gather(dim, index)`, the following will happen:
1. Prepend 1s to the shape of either `t` or `index`, whichever have a smaller
   `.dim()`, to make it the same dimensions.
2. For all dimensions except the one specified by `dim`, expand according to
   the standard rule of broadcasting.
3. The dimension specified by `dim` of `t` and `index` are kept unchanged.

The broadcasting of `scatter` works a bit more complicated than `gather`, as
described below:
Let's say we are doing a `t.scatter_(dim, index, src)`, the following will happen:
1. Prepend 1s to the shape of `index` and/or `src` whichever have a smaller
   `.dim()` than `t`, to make them same dimensions.
2. For all dimensions except the one specified by `dim`, expand `t`, `index`,
   and `src` according to the standard rule of broadcasting. The shape of `t`
   is not allowed to change, which means, if the standard broadcasting rule
   require `t` to be expanded, wes should raise a runtime error.
3. For the dimension specified by `dim`, expand `src` to `index` according to
   the standard rule of broadcasting. Note that `index` will not expand at this
   dimension since `scatter` requires all the values in a row along the this
   dimension must be unique.

*/

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/WrapDimUtils.h"

#include <tuple>
#include <algorithm>

namespace at {
namespace native{

namespace {

std::tuple<std::vector<int64_t>, std::vector<int64_t> > gather_infer_size(IntList self, IntList index, int64_t &dim) {
  auto dimsA = self.size();
  auto dimsB = index.size();
  ptrdiff_t ndim = std::max(dimsA, dimsB);
  dim = maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> expandedSizesB(ndim);
  int64_t expandedSizeA = -1;

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? self[dimA] : 1;
    long sizeB = (dimB >= 0) ? index[dimB] : 1;

    if (i == dim) {
      expandedSizeA = sizeA;
      expandedSizesB[i] = sizeB;
    } else {
      AT_CHECK(
          sizeA == sizeB || sizeA == 1 || sizeB == 1,
          "The size of self (", sizeA,
          ") must match the size of tensor index (", sizeB,
          ") at non-singleton dimension ", i);

      // 1s map to the other size (even 0).
      expandedSizesB[i] = sizeA == 1 ? sizeB : sizeA;
    }
  }

  std::vector<int64_t> expandedSizesA(expandedSizesB);
  if (dim < ndim) expandedSizesA[dim] = expandedSizeA;
  return std::make_tuple(expandedSizesA, expandedSizesB);
}

inline std::tuple<Tensor, Tensor> gather_expand(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name, int64_t &dim) {
  check_defined({to_expand1, to_expand2}, api_name);
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  std::vector<int64_t> expanded_size1, expanded_size2;
  std::tie(expanded_size1, expanded_size2) = gather_infer_size(to_expand1.sizes(), to_expand2.sizes(), dim);
  return std::make_tuple(
      to_expand1.expand(expanded_size1, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size2, /*implicit=*/true));
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> > scatter_infer_size(IntList self, IntList index, IntList src, int64_t &dim) {
  auto dimsA = self.size();
  auto dimsB = index.size();
  auto dimsC = src.size();
  ptrdiff_t ndim = std::max({dimsA, dimsB, dimsC});
  dim = maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> expandedSizesB(ndim);
  std::vector<int64_t> expandedSizesC(ndim);

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long dimC = dimsC - 1 - offset;
    long sizeA = (dimA >= 0) ? self[dimA] : 1;
    long sizeB = (dimB >= 0) ? index[dimB] : 1;
    long sizeC = (dimC >= 0) ? src[dimC] : 1;

    if (i == dim) {
      AT_CHECK(
          sizeB == sizeC || sizeC == 1,
          "The size of tensor index (", sizeB,
          ") must match the size of tensor src (", sizeC,
          ") at non-singleton dimension ", i);
      expandedSizesB[i] = sizeB;
      expandedSizesC[i] = sizeC == 1 ? sizeB : sizeC;
    } else {
      AT_CHECK(
          (sizeA == sizeB && sizeA == sizeC) ||
          (sizeB == 1 && sizeA == sizeC) ||
          (sizeC == 1 && sizeA == sizeB) ||
          (sizeB == 1 && sizeC == 1),
          "The size of tensor self (", sizeA,
          "), tensor index (", sizeB,
          "), and tensor src (", sizeB,
          ") is not broadcastable at non-singleton dimension ", i);

      expandedSizesB[i] = sizeB == 1 ? sizeA : sizeB;
      expandedSizesC[i] = sizeC == 1 ? sizeA : sizeC;
    }
  }

  return std::make_tuple(expandedSizesB, expandedSizesC);
}

inline std::tuple<Tensor, Tensor> scatter_expand(const Tensor &self, const Tensor &index, const Tensor &src, const char *api_name, int64_t &dim) {
  check_defined({self, index, src}, api_name);
  std::vector<int64_t> expanded_size1, expanded_size2;
  std::tie(expanded_size1, expanded_size2) = scatter_infer_size(self.sizes(), index.sizes(), src.sizes(), dim);
  return std::make_tuple(
      index.expand(expanded_size1, /*implicit=*/true), // see [expand implicit]
      src.expand(expanded_size2, /*implicit=*/true));
}

} // namespace


Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor b_self, b_index;
  std::tie(b_self, b_index) = gather_expand(self, index, "gather", dim);
  return at::_s_gather_out(result, b_self, dim, b_index);
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor b_self, b_index;
  std::tie(b_self, b_index) = gather_expand(self, index, "gather", dim);
  return at::_s_gather(b_self, dim, b_index);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  Tensor b_index, b_src;
  std::tie(b_index, b_src) = scatter_expand(self, index, src, "scatter_", dim);
  return self._s_scatter_(dim, b_index, b_src);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar src) {
  Tensor b_index, b_src;
  std::tie(b_index, b_src) = scatter_expand(self, index, self.type().scalarTensor(src), "scatter_", dim);
  return self._s_scatter_(dim, b_index, b_src);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  Tensor b_index, b_src;
  std::tie(b_index, b_src) = scatter_expand(self, index, src, "scatter_add_", dim);
  return self._s_scatter_add_(dim, b_index, b_src);
}

}  // namespace native
}  // namespace at
