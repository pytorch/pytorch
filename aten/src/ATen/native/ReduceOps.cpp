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

using sum_type = void(Tensor &, const Tensor &, size_t, bool);
sum_type *sumImpl = &DispatchStub<sum_type>::init<sumImplC, &sumImpl>;

Tensor _sum_cpu(const Tensor &self) {
  if (self.is_contiguous()) {
    Tensor result = self.type().tensor({});
    sumImpl(result, self, 0, true);
    return result;
  }
  return self._sumall();
}

Tensor _sum_cuda(const Tensor &self_) { return self_._sumall(); }

Tensor sum(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return result;
  }
  // Return identity
  if (self.numel() == 0 && self.ndimension() == 1) {
    result.resize_({0});
    result.fill_(0);
    return result;
  }
  at::sum_out(result, self, dim, keepdim);
  return result;
}

Tensor &_sum_out_cpu(Tensor &result, const Tensor &self, int64_t dim,
                     bool keepdim) {
  if (self.is_contiguous() && result.is_contiguous()) {
    IntList self_sizes = self.sizes();
    std::vector<int64_t> result_sizes;
    result_sizes.insert(result_sizes.end(), self_sizes.begin(),
                        self_sizes.end());

    result_sizes[dim] = 1;
    result.resize_(result_sizes);

    sumImpl(result, self, dim, false);

    if (!keepdim) {
      result.squeeze_(dim);
    }
    return result;
  } else {
    return at::_sum_out(result, self, dim, keepdim);
  }
}

Tensor &_sum_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  return at::_sum_out(result, self, dim, keepdim);
}

using prod_type = void(Tensor &, const Tensor &, size_t, bool);
prod_type *prodImpl = &DispatchStub<prod_type>::init<prodImplC, &prodImpl>;

Tensor _prod_cpu(const Tensor &self) {
  if (self.is_contiguous()) {
    Tensor result = self.type().tensor({});
    prodImpl(result, self, 0, true);
    return result;
  }
  return self._prodall();
}

Tensor _prod_cuda(const Tensor &self_) { return self_._prodall(); }

Tensor prod(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return result;
  }
  // Return identity
  if (self.numel() == 0 && self.ndimension() == 1) {
    result.resize_({0});
    result.fill_(1);
    return result;
  }
  at::prod_out(result, self, dim, keepdim);
  return result;
}

Tensor &_prod_out_cpu(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  if (self.is_contiguous() && result.is_contiguous()) {
    IntList self_sizes = self.sizes();
    std::vector<int64_t> result_sizes;
    result_sizes.insert(result_sizes.end(), self_sizes.begin(),
                        self_sizes.end());

    result_sizes[dim] = 1;
    result.resize_(result_sizes);

    prodImpl(result, self, dim, false);

    if (!keepdim) {
      result.squeeze_(dim);
    }
    return result;
  } else {
    return at::_prod_out(result, self, dim, keepdim);
  }
}

Tensor &_prod_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                       bool keepdim) {
  return at::_prod_out(result, self, dim, keepdim);
}
}
}
