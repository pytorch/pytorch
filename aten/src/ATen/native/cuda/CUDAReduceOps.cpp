#include <ATen/ATen.h>
#include "ATen/native/ReduceOpsUtils.h"

namespace at { namespace native {

Tensor _sum_cuda(const Tensor &self_) { return self_._sumall(); }

Tensor _prod_cuda(const Tensor &self_) { return self_._prodall(); }

Tensor &_sum_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    return at::_th_sum_out(result, self, dim, keepdim);
  }
}

Tensor &_prod_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                       bool keepdim) {
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    return at::_th_prod_out(result, self, dim, keepdim);
  }
}


}}
