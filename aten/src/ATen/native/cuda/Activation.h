#pragma once
#include <cstdint>

namespace at {
struct TensorIteratorBase;
class TensorBase;
}

namespace at { namespace native {

void launch_glu_backward_kernel(const TensorIteratorBase& iter,
                                int64_t gI_stride, int64_t I_stride);

void launch_log_sigmoid_forward_kernel(TensorIteratorBase& iter);

void launch_prelu_cuda_kernel_share_weights(
    TensorIteratorBase &iter, const TensorBase &weight);
void launch_prelu_cuda_kernel_multi_weights(
    const TensorBase &result, const TensorBase &input, const TensorBase &weight);

void launch_prelu_cuda_backward_kernel_share_weights(
    TensorIteratorBase &iter, const TensorBase &weight);
void launch_prelu_cuda_backward_kernel_multi_weights(
    const TensorBase &input, const TensorBase &weight, const TensorBase &grad_out,
    const TensorBase &input_grad, const TensorBase &weight_grad_collector);

void GeluCUDAKernelImpl(TensorIteratorBase& it);
void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it);

}}  // namespace at::native
