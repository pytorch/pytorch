#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "cpu/ReduceOpsKernel.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {
namespace native {

using reduce_type = void(Tensor &, const Tensor &, size_t, bool);
reduce_type *sumImpl = &DispatchStub<reduce_type>::init<sumImplC, &sumImpl>;
reduce_type *prodImpl = &DispatchStub<reduce_type>::init<prodImplC, &prodImpl>;

// ALL REDUCE #################################################################

Tensor _reduce_cpu(reduce_type *f, const Tensor &self) {
  Tensor result = self.type().tensor({});
  f(result, self, 0, true);
  return result;
}

Tensor _sum_cpu(const Tensor &self) {
  if (self.is_contiguous())
    return _reduce_cpu(sumImpl, self);
  return self._sumall();
}

Tensor _prod_cpu(const Tensor &self) {
  if (self.is_contiguous())
    return _reduce_cpu(prodImpl, self);
  return self._prodall();
}

Tensor _sum_cuda(const Tensor &self_) { return self_._sumall(); }

Tensor _prod_cuda(const Tensor &self_) { return self_._prodall(); }

// \ALL REDUCE ################################################################

// DIM REDUCE #################################################################

static bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
                                      int64_t ident) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0 && self.ndimension() == 1) {
    result.resize_({0});
    result.fill_(ident);
    return true;
  }
  return false;
}

static Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntList self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

Tensor &_reduce_out_cpu(reduce_type *f, Tensor &result, const Tensor &self,
                        int64_t dim, bool keepdim) {
  result = _dimreduce_setup(result, self, dim);
  f(result, self, dim, false);
  if (!keepdim)
    result.squeeze_(dim);
  return result;
}

Tensor &_sum_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                     bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 0))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    return _reduce_out_cpu(sumImpl, result, self, dim, keepdim);
  }
  return at::_sum_out(result, self, dim, keepdim);
}

Tensor &_prod_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                      bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 1))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    return _reduce_out_cpu(prodImpl, result, self, dim, keepdim);
  }
  return at::_prod_out(result, self, dim, keepdim);
}

Tensor &_sum_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  return at::_sum_out(result, self, dim, keepdim);
}

Tensor &_prod_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                       bool keepdim) {
  return at::_prod_out(result, self, dim, keepdim);
}

Tensor sum(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::sum_out(result, self, dim, keepdim);
}

Tensor prod(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::prod_out(result, self, dim, keepdim);
}

// \DIM REDUCE ################################################################
}
}
