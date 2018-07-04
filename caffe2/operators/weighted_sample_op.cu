#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/weighted_sample_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

__global__ void WeightedSampleKernel(
    const int batch_size,
    const int weights_dim,
    const float* in_weights_data,
    const float* in_val_data,
    float* samples,
    int* out_idx_data,
    float* out_val_data) {
  CUDA_1D_KERNEL_LOOP(i, batch_size) {
    int offset = i * weights_dim;

    float sum = 0.0;
    for (int j = 0; j < weights_dim; j++) {
      sum += in_weights_data[offset + j];
    }
    samples[i] *= sum;

    float cum_sum = 0.0;
    int j = 0;
    for (; j < weights_dim; j++) {
      cum_sum += in_weights_data[offset + j];
      if (cum_sum >= samples[i]) {
        break;
      }
    }
    out_idx_data[i] = min(j, weights_dim - 1);

    if (out_val_data) {
      out_val_data[i] = in_val_data[offset + out_idx_data[i]];
    }
  }
}

} // namespace

template <>
bool WeightedSampleOp<float, CUDAContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(
      InputSize(),
      OutputSize(),
      "The number of tensors of the input and the output must be the same.");

  auto& in_weights = Input(0);
  auto* out_idx = Output(0);
  int batch_size = in_weights.dim(0);
  int weights_dim = in_weights.dim(1);

  if (batch_size > 0 && weights_dim > 0) {
    out_idx->Resize(batch_size, 1);
    unif_samples_.Resize(batch_size);

    const float* in_weights_data = in_weights.data<float>();
    const float* in_val_data = nullptr;
    int* out_idx_data = out_idx->mutable_data<int>();
    float* out_val_data = nullptr;

    if (OutputSize() == 2) {
      auto& in_val = Input(1);
      CAFFE_ENFORCE_EQ(
          in_weights.dims(),
          in_val.dims(),
          "The sampling weights tensor and the sampling values tensor must have the same dimensions.");
      in_val_data = in_val.data<float>();

      auto* out_val = Output(1);
      out_val->Resize(batch_size, 1);
      out_val_data = out_val->mutable_data<float>();
    }

    float* unif_samples_data = unif_samples_.mutable_data<float>();
    CURAND_ENFORCE(curandGenerateUniform(
        context_.curand_generator(), unif_samples_data, batch_size));

    WeightedSampleKernel<<<
        CAFFE_GET_BLOCKS(batch_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        batch_size,
        weights_dim,
        in_weights_data,
        in_val_data,
        unif_samples_data,
        out_idx_data,
        out_val_data);
  } else {
    out_idx->Resize(0);
    out_idx->mutable_data<int>();
    if (OutputSize() == 2) {
      auto* out_val = Output(1);
      out_val->Resize(0);
      out_val->mutable_data<float>();
    }
  }

  return true;
}

REGISTER_CUDA_OPERATOR(WeightedSample, WeightedSampleOp<float, CUDAContext>);
} // namespace caffe2
