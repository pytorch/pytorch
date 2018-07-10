#include "caffe2/sgd/rmsprop_op.h"
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

__global__ void RmsPropUpdate(
    int N,
    const float* g,
    const float* ms,
    const float* mom,
    float* ng,
    float* nms,
    float* nmom,
    float decay,
    float momentum,
    float epsilon,
    const float* lr) {
  HIP_1D_KERNEL_LOOP(i, N) {
    // Update new mean square estimate
    nms[i] = ms[i] + (1.0f - decay) * (g[i] * g[i] - ms[i]);
    // Update momentum estimate
    nmom[i] =
        mom[i] * momentum + lr[0] * g[i] / sqrtf(epsilon + nms[i]);
    // New gradient is the momentum
    ng[i] = nmom[i];
  }
}

template <>
void rmsprop_update<HIPContext>(
    int N,
    const float* g,
    const float* ms,
    const float* mom,
    float* ng,
    float* nms,
    float* nmom,
    float decay,
    float momentum,
    float epsilon,
    const float* lr,
    HIPContext* context) {
  hipLaunchKernelGGL(RmsPropUpdate, dim3(CAFFE_GET_BLOCKS(static_cast<int>(N))), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), 
      static_cast<int>(N), g, ms, mom, ng, nms, nmom, decay, momentum, epsilon, lr);
}


REGISTER_HIP_OPERATOR(RmsProp, RmsPropOp<float, HIPContext>);

}
