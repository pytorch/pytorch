#pragma once
#include <ATen/native/Activation.h>
#include <cstdint>

namespace at {
struct TensorIteratorBase;
class TensorBase;
}

namespace at { namespace native {

void launch_glu_backward_kernel(const TensorIteratorBase& iter,
                                int64_t gI_stride, int64_t I_stride);

void launch_log_sigmoid_forward_kernel(TensorIteratorBase& iter);

void GeluCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate);
void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate);

}}  // namespace at::native
