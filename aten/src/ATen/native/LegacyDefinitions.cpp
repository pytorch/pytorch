#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

int64_t storage_offset(const Tensor& self) {
  return self._th_storage_offset();
}

int64_t ndimension(const Tensor& self) {
  return self._th_ndimension();
}

Tensor & resize_(Tensor& self, IntList size) {
  return self._th_resize_(size);
}

Tensor & set_(Tensor& self, Storage source) {
  return self._th_set_(source);
}

Tensor & set_(Tensor& self, Storage source, int64_t storage_offset, IntList size, IntList stride) {
  return self._th_set_(source, storage_offset, size, stride);
}

Tensor & set_(Tensor& self, const Tensor & source) {
  return self._th_set_(source);
}

Tensor & set_(Tensor& self) {
  return self._th_set_();
}

bool is_contiguous(const Tensor& self) {
  return self._th_is_contiguous();
}

bool is_set_to(const Tensor& self, const Tensor & tensor) {
  return self._th_is_set_to(tensor);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, Scalar value) {
  return self._th_masked_fill_(mask, value);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, const Tensor & value) {
  return self._th_masked_fill_(mask, value);
}

Tensor & masked_scatter_(Tensor& self, const Tensor & mask, const Tensor & source) {
  return self._th_masked_scatter_(mask, source);
}

Tensor view(const Tensor& self, IntList size) {
  return self._th_view(size);
}

Tensor & digamma_(Tensor& self) {
  return self._th_digamma_();
}

Tensor & polygamma_(Tensor& self, int64_t n) {
  return self._th_polygamma_(n);
}

Tensor & erfinv_(Tensor& self) {
  return self._th_erfinv_();
}

Tensor & frac_(Tensor& self) {
  return self._th_frac_();
}

Tensor & renorm_(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  return self._th_renorm_(p, dim, maxnorm);
}

Tensor & reciprocal_(Tensor& self) {
  return self._th_reciprocal_();
}

Tensor & neg_(Tensor& self) {
  return self._th_neg_();
}

Tensor & pow_(Tensor& self, Scalar exponent) {
  return self._th_pow_(exponent);
}

Tensor & pow_(Tensor& self, const Tensor & exponent) {
  return self._th_pow_(exponent);
}

Tensor & lerp_(Tensor& self, const Tensor & end, Scalar weight) {
  return self._th_lerp_(end, weight);
}

Tensor & sign_(Tensor& self) {
  return self._th_sign_();
}

Tensor & fmod_(Tensor& self, Scalar other) {
  return self._th_fmod_(other);
}

Tensor & fmod_(Tensor& self, const Tensor & other) {
  return self._th_fmod_(other);
}

Tensor & remainder_(Tensor& self, Scalar other) {
  return self._th_remainder_(other);
}

Tensor & remainder_(Tensor& self, const Tensor & other) {
  return self._th_remainder_(other);
}

Tensor & addbmm_(Tensor& self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return self._th_addbmm_(batch1, batch2, beta, alpha);
}

Tensor & addcmul_(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return self._th_addcmul_(tensor1, tensor2, value);
}

Tensor & addcdiv_(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return self._th_addcdiv_(tensor1, tensor2, value);
}

Tensor & random_(Tensor& self, int64_t from, int64_t to, Generator * generator) {
  return self._th_random_(from, to, generator);
}

Tensor & random_(Tensor& self, int64_t to, Generator * generator) {
  return self._th_random_(to, generator);
}

Tensor & random_(Tensor& self, Generator * generator) {
  return self._th_random_(generator);
}

Tensor & uniform_(Tensor& self, double from, double to, Generator * generator) {
  return self._th_uniform_(from, to, generator);
}

Tensor & normal_(Tensor& self, double mean, double std, Generator * generator) {
  return self._th_normal_(mean, std, generator);
}

Tensor & cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  return self._th_cauchy_(median, sigma, generator);
}

Tensor & log_normal_(Tensor& self, double mean, double std, Generator * generator) {
  return self._th_log_normal_(mean, std, generator);
}

Tensor & exponential_(Tensor& self, double lambd, Generator * generator) {
  return self._th_exponential_(lambd, generator);
}

Tensor & geometric_(Tensor& self, double p, Generator * generator) {
  return self._th_geometric_(p, generator);
}

}} // namespace at::native
