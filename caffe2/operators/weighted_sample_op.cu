#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/weighted_sample_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

__global__ void WeightedSampleKernel(
    const int N,
    const float* mat_weights,
    int weights_dim,
    float* samples,
    int* output_indices) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int offset = i * weights_dim;
    float cum_sum = 0.0;
    int j = 0;
    for (; j < weights_dim; j++) {
      cum_sum += mat_weights[offset + j];
      if (cum_sum >= samples[i]) {
        break;
      }
    }
    output_indices[i] = min(j, weights_dim - 1);
  }
}

} // namespace

template <>
bool WeightedSampleOp<float, CUDAContext>::RunOnDevice() {
  auto& weights = Input(0);
  int batch_size = weights.dim(0);
  int weights_dim = weights.dim(1);
  auto* output = Output(0);

  if (batch_size > 0 && weights_dim > 0) {
    output->Resize(batch_size, 1);
    if (batch_size != unif_samples_.size()) {
      unif_samples_.Resize(batch_size);
    }
    const float* mat_weights = weights.data<float>();
    int* output_indices = output->template mutable_data<int>();
    float* unif_samples_data = unif_samples_.mutable_data<float>();

    CURAND_ENFORCE(curandGenerateUniform(
        context_.curand_generator(), unif_samples_data, batch_size));

    WeightedSampleKernel<<<
        CAFFE_GET_BLOCKS(batch_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        batch_size,
        mat_weights,
        weights_dim,
        unif_samples_data,
        output_indices);
  } else {
    output->Resize(0);
    output->template mutable_data<int>();
  }

  return true;
}

REGISTER_CUDA_OPERATOR(WeightedSample, WeightedSampleOp<float, CUDAContext>);
} // namespace caffe2
