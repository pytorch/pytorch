#include <cfloat>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "softmax_op.h"
#include "softmax_with_loss_op.h"
#include "spatial_softmax_with_loss_op.h"

namespace caffe2 {

namespace {

__global__ void LabelCrossEntropyKernel(
    const int N,
    const int D,
    const float* logPdata,
    const int* labeldata,
    const float* weights,
    float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    CUDA_KERNEL_ASSERT(labeldata[i] >= 0 && labeldata[i] < D);
    float weight = weights ? weights[i] : 1.0;
    Ydata[i] = -logPdata[i * D + labeldata[i]] * weight;
  }
}

__global__ void LabelCrossEntropyGradientKernel(
    const int N,
    const int D,
    const float* Pdata,
    const int* labeldata,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int idx = i * D + labeldata[i];
    dXdata[idx] = Pdata[idx] - 1.;
  }
}

__global__ void LabelCrossEntropyGradientKernelWeighted(
    const int N,
    const int D,
    const float* Pdata,
    const int* labeldata,
    float* dXdata,
    const float* weights) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    int row = i / D;
    int d = i % D;
    float val = Pdata[i] - 1.0 * (d == labeldata[row]);
    float weight = weights[row];
    dXdata[i] = val * weight;
  }
}

__global__ void ProbCrossEntropyKernel(
    const int N,
    const int D,
    const float* Pdata,
    const float* labeldata,
    const float* weights,
    float* Ydata) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float weight = weights ? weights[i] : 1.0;
    float sum = 0.0;
    float total_prob = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      int idx = i * D + j;
      CUDA_KERNEL_ASSERT(labeldata[idx] >= 0);
      total_prob += labeldata[idx];
      sum += -logf(max(Pdata[idx], FLT_MIN)) * labeldata[idx] * weight;
    }
    float tot = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    float total_prob_sum = BlockReduce(temp_storage).Sum(total_prob);
    if (threadIdx.x == 0) {
      Ydata[i] = tot;
      // Sanity check
      CUDA_KERNEL_ASSERT(abs(1.0 - total_prob_sum) < 1e-5f);
    }
    __syncthreads();
  }
}

__global__ void ProbCrossEntropyGradientKernel(
    const int N,
    const int D,
    const float* Pdata,
    const float* labeldata,
    float* dXdata,
    const float* weights) {
  if (weights == NULL) {
    CUDA_1D_KERNEL_LOOP(idx, N * D) {
      dXdata[idx] = Pdata[idx] - labeldata[idx];
    }
  } else {
    CUDA_1D_KERNEL_LOOP(idx, N * D) {
      dXdata[idx] = (Pdata[idx] - labeldata[idx]) * weights[idx / D];
    }
  }
}

__global__ void SpatialSoftmaxKernel(
    const int num,
    const int D,
    const int W,
    const int H,
    const float* Xdata,
    float* Pdata) {
  CUDA_1D_KERNEL_LOOP(index, num * W * H) {
    int x = index % W;
    int y = (index / W) % H;
    int i = index / W / H;

    // Subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c = 0; c < D; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      max_val = max(max_val, Xdata[idx]);
    }

    // Exponentiate
    float expsum = 0.0f;
    for(int c = 0; c < D; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      float expx = exp(Xdata[idx] - max_val);
      Pdata[idx] = expx;
      expsum += expx;
    }

    // Normalize
    for(int c=0; c<D; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      Pdata[idx] /= expsum;
    }
  }
}


#define DONTCARE (-1)

__global__ void SpatialCrossEntropyLossKernel(
    const int N,
    const int D,
    const int W,
    const int H,
    const float* Pdata,
    const int* label_data,
    const float* weights,
    float* loss_data,
    float* weight_data) {
  CUDA_1D_KERNEL_LOOP(index, N * W * H) {
    int x = index % W;
    int y = (index / W) % H;
    int i = index / W / H;
    const int label = static_cast<int>(label_data[index]);

    if (label != DONTCARE) {
      CUDA_KERNEL_ASSERT(label >= 0 && label < D);
      float weight = (weights == NULL ? 1.0 : weights[index]);
      loss_data[index] = -log(max(
        Pdata[i * W * H * D + label * W * H + y * W + x], 1e-20f)) * weight;
      weight_data[index] = weight;
    } else {
      loss_data[index] = 0;
      weight_data[index] = 0;
    }
  }
}

__global__ void SpatialSoftmaxLossGradientKernel(const int N, const int D,
    const int W, const int H, const int* label_data, const float* weights,
         float* dX_data, float* weights_) {
 CUDA_1D_KERNEL_LOOP(index, N * W * H) {
   int x = index % W;
   int y = (index / W) % H;
   int i = index / W / H;
   const int label = static_cast<int>(label_data[index]);

   if (label != DONTCARE) {
     int data_idx = i * (H * W * D) + label * (H * W) + y * W + x;
     dX_data[data_idx] -= 1.0;
     if (weights != NULL) {
       float weight = weights[index];
       for (int c = 0; c < D; ++c) {
         int data_idx = i * (H * W * D) + c * (H * W) + y * W + x;
         dX_data[data_idx] *= weight;
       }
       weights_[index] = weight;
     } else {
       weights_[index] = 1.0;
     }
   } else {
     // Ignore-label, so set all gradients for this positions
     // tp zero
     for (int c = 0; c < D; ++c) {
       int data_idx = i * (H * W * D) + c * (H * W) + y * W + x;
       dX_data[data_idx] = 0.0;
     }
     weights_[index] = 0.0;
   }
 }
}

__global__ void SoftmaxNormalizeLogsKernel(
    const int nthreads,
    const int D,
    const float* logits,
    const float* rowmax,
    const float* scales,
    float* out_log) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out_log[index] = logits[index] - rowmax[n] - logf(max(scales[n], FLT_MIN));
  }
}

__global__ void SoftmaxNormalizeKernel(
    const int nthreads,
    const int D,
    const float* probs,
    const float* scales,
    float* out) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out[index] = probs[index] / scales[n];
  }
}

void Softmax(
    const int N,
    const int D,
    const float* logits,
    const float* sum_multiplier,
    float* scales,
    float* rowmax,
    float* probs,
    bool log_softmax,
    CUDAContext* context) {
  const int size = N * D;

  math::RowwiseMax<float, CUDAContext>(N, D, logits, rowmax, context);
  // Put the intermediate result X - max(X) into Y
  context->Copy<float, CUDAContext, CUDAContext>(size, logits, probs);
  // Subtract the scale
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      rowmax,
      sum_multiplier,
      1,
      probs,
      context);
  // Exponentiation
  math::Exp<float, CUDAContext>(size, probs, probs, context);
  // Sum exponentiated values
  math::Gemv<float, CUDAContext>(CblasNoTrans, N, D, 1, probs, sum_multiplier,
                                 0, scales, context);
  // Normalize
  if (!log_softmax) {
    SoftmaxNormalizeKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, probs, scales, probs);
  } else {
    SoftmaxNormalizeLogsKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, logits, rowmax, scales, probs);
  }
}

} // namespace

template<>
bool SoftmaxWithLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto& T = Input(1);  // Labels / targets
  auto* P = Output(0); // Probabilities from softmax
  auto* avg_loss = Output(1); // Average loss
  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : NULL);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  P->ResizeLike(X);
  total_weight_ptr_.Resize(1);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.ndim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.ndim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.size(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  avg_loss->Resize(vector<TIndex>());
  if (losses_.size() != N) {
    losses_.Resize(N);
  }
  if (rowmax_.size() != N) {
    rowmax_.Resize(N);
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }
  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      losses_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P->mutable_data<float>(),
      !label_prob_mode_, // logarithmic output
      &context_);
  // Compute label xent loss per example
  if (!label_prob_mode_) {
    LabelCrossEntropyKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P->data<float>(),
        T.data<int>(),
        weights,
        losses_.mutable_data<float>());
    // Since we had logarithmic output, we need to exponentiate
    // them again.
    math::Exp<float, CUDAContext>(
        N * D, P->data<float>(), P->mutable_data<float>(), &context_);
  } else {
    ProbCrossEntropyKernel<<<
        std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P->data<float>(),
        T.data<float>(),
        weights,
        losses_.mutable_data<float>());
  }

  float total_weight = N;
  if (weights) {
    // Sum weights
    math::Sum<float, CUDAContext>(
        N, weights, total_weight_ptr_.mutable_data<float>(), &context_, &scratch_);
    cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream());
  }

  // Sum of all losses
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_, &scratch_);
  // Average of input batch size
  if (total_weight > 0) {
    math::Scale<float, CUDAContext>(
        1, scale_ / total_weight, avg_loss_data, avg_loss_data, &context_);
  }

  return true;
}

template <>
bool SpatialSoftmaxWithLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets
  auto* P = Output(0); // Probabilities from softmax
  auto* avg_loss = Output(1); // Average loss
  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : NULL);
  int N, D;
  N = X.dim32(0);
  D = X.dim32(1);
  P->ResizeLike(X);
  total_weight_ptr_.Resize(1);
  CAFFE_ENFORCE_EQ(X.ndim(), 4);
  CAFFE_ENFORCE_EQ(T.ndim(), 3);
  CAFFE_ENFORCE_EQ(T.dim32(0), N);

  int H = X.dim32(2);
  int W = X.dim32(3);
  if (losses_.size() != N * W * H) {
    losses_.Resize(N * W * H);
  }
  if (weights_.size() != N * W * H) {
    weights_.Resize(N * W * H);
  }

  const float* Xdata = X.data<float>();
  float* Pdata = P->mutable_data<float>();

  // Softmax for each x,y location
  SpatialSoftmaxKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, W, H, Xdata, Pdata);

  // Cross entropy
  avg_loss->Resize(vector<TIndex>());
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Set<float, CUDAContext>(1, 0.0f, avg_loss_data, &context_);

  const int* label_data = T.data<int>();
  math::Set<float, CUDAContext>(
      1, 0.0f, total_weight_ptr_.mutable_data<float>(), &context_);

  SpatialCrossEntropyLossKernel<<<
      CAFFE_GET_BLOCKS(N * W * H),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      W,
      H,
      P->data<float>(),
      label_data,
      weights,
      losses_.mutable_data<float>(),
      weights_.mutable_data<float>());

  // Somewhat awkward scalar passing from device to host
  float h_total_weight;
  math::Sum<float, CUDAContext>(
      weights_.size(),
      weights_.data<float>(),
      total_weight_ptr_.mutable_data<float>(),
      &context_,
      &scratch_);
  cudaMemcpyAsync(
      &h_total_weight,
      total_weight_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream());

  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_, &scratch_);

  // Final scaling
  if (h_total_weight > 0) {
    math::Scale<float, CUDAContext>(
        1, scale_ / h_total_weight, avg_loss_data, avg_loss_data, &context_);
  }

  return true;
}

template <>
bool SoftmaxWithLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto& T = Input(1);  // Labels / targets
  // Input(2) is weights, if given
  auto& P = Input(InputSize() - 2);  // Probabilities from softmax
  auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss
  const float* weights = (InputSize() > 4 ? Input(2).data<float>() : NULL);

  auto* dX = Output(0);
  dX->ResizeLike(X);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);

  if (only_loss_) {
    // Memory saving trick to share the buffer with the softmax output.
    // Softmax output is thus overwritten.
    dX->ShareData(P);
  }

  total_weight_ptr_.Resize(1);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.ndim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.ndim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.size(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  // Subtract 1 from labeled positions
  if (!label_prob_mode_) {
    if (weights == nullptr) {
      // Copy softmax probabilities into dX
      if (!only_loss_) {
        context_.Copy<float, CUDAContext, CUDAContext>(
            P.size(), P.data<float>(), dX->mutable_data<float>());
      }
      LabelCrossEntropyGradientKernel<<<
          CAFFE_GET_BLOCKS(N),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N, D, P.data<float>(), T.data<int>(), dX->mutable_data<float>());
    } else {
      // Weighted version gets the Pdata values internally
      LabelCrossEntropyGradientKernelWeighted<<<
          CAFFE_GET_BLOCKS(N * D),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          D,
          P.data<float>(),
          T.data<int>(),
          dX->mutable_data<float>(),
          weights);
    }
  } else {
    ProbCrossEntropyGradientKernel<<<
        CAFFE_GET_BLOCKS(N * D),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P.data<float>(),
        T.data<float>(),
        dX->mutable_data<float>(),
        weights);
  }
  float total_weight = N;
  if (weights) {
    // Sum weights
    math::Sum<float, CUDAContext>(
        N, weights, total_weight_ptr_.mutable_data<float>(), &context_, &scratch_);
    cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream());
  }

  // Scale by d_avg_loss / N
  if (total_weight > 0) {
    math::Scale<float, CUDAContext>(
        dX->size(),
        scale_ / total_weight,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  }
  math::Scale<float, CUDAContext>(
      dX->size(),
      d_avg_loss.data<float>(),
      dX->data<float>(),
      dX->mutable_data<float>(),
      &context_);

  return true;
}

template <>
bool SpatialSoftmaxWithLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets
  // Input(2) is weights, if given
  auto& P = Input(InputSize() - 2); // Probabilities from softmax
  auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss
  const float* weights = (InputSize() > 4 ? Input(2).data<float>() : NULL);

  auto* dX = Output(0);
  dX->ResizeLike(X);

  const auto canonical_axis = X.canonical_axis_index(1);
  int N, D;
  N = X.dim32(0);
  D = X.dim32(1);

  if (only_loss_) {
    // Memory saving trick to share the buffer with the softmax output.
    // Softmax output is thus overwritten.
    dX->ShareData(P);
  }

  total_weight_ptr_.Resize(1);
  // Spatial mode, compute softmax for each x, y location
  CAFFE_ENFORCE_EQ(X.ndim(), 4);
  CAFFE_ENFORCE_EQ(T.ndim(), 3);

  int H = X.dim32(2);
  int W = X.dim32(3);
  dX->ResizeLike(X);
  if (weights_.size() != N * W * H) {
    weights_.Resize(N * W * H);
  }

  const float* Pdata = P.data<float>();
  float* dX_data = dX->mutable_data<float>();
  const int* label_data = T.data<int>();
  const float* d_avg_loss_data = d_avg_loss.data<float>();

  // Copy softmax probabilities into dX. All but the neuron
  // corresponding to the correct label has gradient equaling e(x_j)
  // which is the probability under softmax.
  context_.Copy<float, CUDAContext, CUDAContext>(P.size(), Pdata, dX_data);

  math::Set<float, CUDAContext>(
      1, 0.0f, total_weight_ptr_.mutable_data<float>(), &context_);

  SpatialSoftmaxLossGradientKernel<<<
      CAFFE_GET_BLOCKS(N * W * H),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, W, H, label_data, weights, dX_data, weights_.mutable_data<float>());

  math::Sum<float, CUDAContext>(
      weights_.size(),
      weights_.data<float>(),
      total_weight_ptr_.mutable_data<float>(),
      &context_,
      &scratch_);

  // Somewhat awkward scalar passing from device to host
  float h_total_weight;
  cudaMemcpyAsync(
      &h_total_weight,
      total_weight_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream());

  // Final scaling
  if (h_total_weight > 0) {
    math::Scale<float, CUDAContext>(
        dX->size(),
        scale_ / h_total_weight,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  }
  math::Scale<float, CUDAContext>(
      dX->size(),
      d_avg_loss.data<float>(),
      dX->data<float>(),
      dX->mutable_data<float>(),
      &context_);

  return true;
}

// Implementation for the CUDA context.
template <>
bool SoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* P = Output(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  P->ResizeLike(X);
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }
  if (scale_.size() != N) {
    scale_.Resize(N);
  }
  if (rowmax_.size() != N) {
    rowmax_.Resize(N);
  }
  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      scale_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P->mutable_data<float>(),
      false,
      &context_);
  return true;
}
#define SOFTMAX_NUM_THREADS 128

// The softmax gradient kernel. This kernel has to be called with the number of
// threads per block being no more than SOFTMAX_NUM_THREADS.
namespace {
__global__ void softmax_gradient_kernel(
    const int dim,
    const float* Y,
    const float* dY,
    float* dX) {
  Y += blockIdx.x * dim;
  dY += blockIdx.x * dim;
  dX += blockIdx.x * dim;
  const int idx = threadIdx.x;
  __shared__ float reduction_buffer[SOFTMAX_NUM_THREADS];
  float tmp;

  // A two-level reduction to compute the inner products.
  tmp = 0;
  for (int i = idx; i < dim; i += blockDim.x) {
    tmp += dY[i] * Y[i];
  }
  reduction_buffer[idx] = tmp;
  __syncthreads();
  if (idx == 0) {
    tmp = reduction_buffer[0];
    for (int i = 1; i < blockDim.x; ++i)
      tmp += reduction_buffer[i];
    reduction_buffer[0] = tmp;
  }
  __syncthreads();
  // Compute gradient.
  tmp = reduction_buffer[0];
  for (int i = idx; i < dim; i += blockDim.x) {
    dX[i] = Y[i] * (dY[i] - tmp);
  }
}
} // namespace

template <>
bool SoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  const auto canonical_axis = Y.canonical_axis_index(axis_);
  const int N = Y.size_to_dim(canonical_axis);
  const int D = Y.size_from_dim(canonical_axis);
  dX->ResizeLike(Y);
  softmax_gradient_kernel<<<
      N,
      SOFTMAX_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      D, Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(SoftmaxWithLoss,
                       SoftmaxWithLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxWithLossGradient,
                       SoftmaxWithLossGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SpatialSoftmaxWithLoss,
    SpatialSoftmaxWithLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SpatialSoftmaxWithLossGradient,
    SpatialSoftmaxWithLossGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Softmax, SoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CUDAContext>);

} // namespace caffe2
