#include <ATen/ATen.h>

namespace at { namespace native {

Tensor _sum_cuda(const Tensor &self_) { return self_._sumall(); }

Tensor _prod_cuda(const Tensor &self_) { return self_._prodall(); }

Tensor &_sum_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  return at::_th_sum_out(result, self, dim, keepdim);
}

Tensor &_prod_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                       bool keepdim) {
  return at::_th_prod_out(result, self, dim, keepdim);
}


}}
