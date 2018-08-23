#include "ATen/ATen.h"

namespace at { namespace native {

void pdist_kernel_cuda(Tensor& result, const Tensor& self, double p);

void pdist_backward_kernel_cuda(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist);

}} // at::native
