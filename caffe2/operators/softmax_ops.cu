#include <cfloat>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <limits>
#include <numeric>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/random_softmax_op.h"
#include "caffe2/operators/softmax_op.h"
#include "caffe2/operators/softmax_with_loss_op.h"
#include "caffe2/operators/spatial_softmax_with_loss_op.h"

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
      sum += -logf(fmaxf(Pdata[idx], FLT_MIN)) * labeldata[idx] * weight;
    }
    float tot = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    float total_prob_sum = BlockReduce(temp_storage).Sum(total_prob);
    if (threadIdx.x == 0) {
      Ydata[i] = tot;
      // Sanity check
      CUDA_KERNEL_ASSERT(fabsf(1.0 - total_prob_sum) < 1e-5f);
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
      max_val = fmaxf(max_val, Xdata[idx]);
    }

    // Exponentiate
    float expsum = 0.0f;
    for(int c = 0; c < D; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      float expx = expf(Xdata[idx] - max_val);
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
      loss_data[index] = -logf(fmaxf(
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
    out_log[index] = logits[index] - rowmax[n] - logf(fmaxf(scales[n], FLT_MIN));
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

// Input P: rows * cols size, label: rows * cols size
// Output Y: size rows * 1,
// where Y[i] = sum_j(-log(P[i][j]) * label[i][j])
__global__ void CrossEntropyKernel(
    const int N,
    const int D,
    const float* P,
    const float* label,
    float* Y) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float sum = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      int idx = i * D + j;
      sum += -logf(fmaxf(P[idx], FLT_MIN)) * label[idx];
    }
    float tot = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    if (threadIdx.x == 0) {
      Y[i] = tot;
    }
    __syncthreads();
  }
}

// Count number of positive labels per training example
// Input label_data: rows * cols size,
// Output num_positive_labels: size rows * 1,
// which is # of label_data == 1.0f per training example
__global__ void RowwiseCountPositiveLabelsKernel(
    const int rows,
    const int cols,
    const float* label_data,
    int* num_positive_labels) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    int count = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      int idx = i * cols + j;
      if (label_data[idx] >= 0.5f) {
        count += 1;
      }
    }
    count = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) {
      num_positive_labels[i] = count;
    }
    __syncthreads();
  }
}

// For each training example, sample K classes:
//    1. keep all classes where label is positive, and
//    2. randomly sample K - num_positive_labels
//      out of D - num_positive_labels negative label classes
// based on Knuth selection sampling technique
// (Algorithm S, The Art of Computer Programming 3.4.2)
// Input:
//    label_data, size N * D,
//    rand_data, size N * D,
//    num_postive_labels, size N * 1,
//      which stores num positive labels for each training example
// Output: sample, size N * K, which stores classes that are sampled
__global__ void RandomSoftmaxSamplingKernel(
    const int N, // batch size
    const int D, // num all classes
    const int K, // num classes to sample
    const float* rand_data,
    const float* label_data,
    const int* num_positive_labels,
    int* samples) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int offset = i * D;
    int t = 0; // total negative labels dealt with
    int m = 0; // number of negative labels items selected so far
    int j = 0; // number of labels visited
    int k = num_positive_labels[i]; // number of positive labels
    int k0 = 0; // num true labels selected so far
    while (m < K - k) {
      // classes which have positive label are always sampled
      if (label_data[offset + j] >= 0.5f) {
        samples[i * K + k0] = j;
        k0++;
      } else {
        // pool size: D - k (total num negatives), sample size: K - k
        float u = rand_data[offset + j];
        if ((D - k - t) * u < K - k - m) {
          samples[i * K + k + m] = j;
          m++;
        }
        t++;
      }
      j++;
    }

    // after K - k negative label classes has been sampled,
    // continue visiting the row to select the rest of positive label classes
    while (j < D && k0 < k) {
      if (label_data[offset + j] >= 0.5f) {
        samples[i * K + k0] = j;
        k0++;
      }
      j++;
    }
  }
}

// Sample the input, of size N * D
// based on sampled indices int* samples, size N * K
// Output is of size N * K,
__global__ void SampleInputKernel(
    const int N,
    const int D,
    const int K,
    const int* samples,
    const float* in,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, N * K) {
    CUDA_KERNEL_ASSERT(samples[i] >= 0 && samples[i] < D);
    int row = i / K;
    out[i] = in[row * D + samples[i]];
  }
}

__global__ void SampleOutputKernel(
    const int N,
    const int D,
    const int K,
    const int* samples,
    const float* in,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, N * K) {
    int row = i / K;
    out[row * D + samples[i]] = in[i];
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
  context->CopySameDevice<float>(size, logits, probs);
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

  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : NULL);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);

  auto* P =
      Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax
  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));
  total_weight_ptr_.Resize(1);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  auto* avg_loss =
      Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss
  if (!losses_.defined()) {
    losses_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (losses_.numel() != N) {
    losses_.Resize(N);
  }

  if (!rowmax_.defined()) {
    rowmax_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (rowmax_.numel() != N) {
    rowmax_.Resize(N);
  }

  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      losses_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P->template mutable_data<float>(),
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
        N * D, P->data<float>(), P->template mutable_data<float>(), &context_);
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
    CUDA_CHECK(cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream()));
  }

  // Sum of all losses
  float* avg_loss_data = avg_loss->template mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.numel(), losses_.data<float>(), avg_loss_data, &context_, &scratch_);
  // Average of input batch size
  if (total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
        1, scale_ / total_weight, avg_loss_data, avg_loss_data, &context_);
  }

  return true;
}

template <>
bool SpatialSoftmaxWithLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets

  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : NULL);
  int N, D;
  N = X.dim32(0);
  D = X.dim32(1);

  auto* P =
      Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax
  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));

  CAFFE_ENFORCE_EQ(X.dim(), 4);
  CAFFE_ENFORCE_EQ(T.dim(), 3);
  CAFFE_ENFORCE_EQ(T.dim32(0), N);

  int H = X.dim32(2);
  int W = X.dim32(3);
  if (!losses_.defined()) {
    losses_ = caffe2::empty({N * W * H}, at::dtype<float>().device(CUDA));
  } else if (losses_.numel() != N * W * H) {
    losses_.Resize(N * W * H);
  }

  if (!weights_.defined()) {
    weights_ = caffe2::empty({N * W * H}, at::dtype<float>().device(CUDA));
  } else if (weights_.numel() != N * W * H) {
    weights_.Resize(N * W * H);
  }

  const float* Xdata = X.data<float>();
  float* Pdata = P->template mutable_data<float>();

  // Softmax for each x,y location
  SpatialSoftmaxKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, W, H, Xdata, Pdata);

  // Cross entropy
  auto* avg_loss =
      Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss
  float* avg_loss_data = avg_loss->template mutable_data<float>();
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
      weights_.numel(),
      weights_.data<float>(),
      total_weight_ptr_.mutable_data<float>(),
      &context_,
      &scratch_);
  CUDA_CHECK(cudaMemcpyAsync(
      &h_total_weight,
      total_weight_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream()));

  math::Sum<float, CUDAContext>(
      losses_.numel(), losses_.data<float>(), avg_loss_data, &context_, &scratch_);

  // Final scaling
  if (h_total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
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

  Tensor* dX;
  if (only_loss_) {
    // Memory saving trick to share the buffer with the softmax output.
    // Softmax output is thus overwritten.
    dX = OutputTensorAlias(0, P);
    dX->ResizeLike(X);
  } else {
    dX = Output(0, X.sizes(), at::dtype<float>());
  }

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);

  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
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
        context_.CopySameDevice<float>(
            P.numel(), P.data<float>(), dX->template mutable_data<float>());
      }
      LabelCrossEntropyGradientKernel<<<
          CAFFE_GET_BLOCKS(N),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          D,
          P.data<float>(),
          T.data<int>(),
          dX->template mutable_data<float>());
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
          dX->template mutable_data<float>(),
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
        dX->template mutable_data<float>(),
        weights);
  }
  float total_weight = N;
  if (weights) {
    // Sum weights
    math::Sum<float, CUDAContext>(
        N, weights, total_weight_ptr_.mutable_data<float>(), &context_, &scratch_);
    CUDA_CHECK(cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream()));
  }

  // Scale by d_avg_loss / N
  if (total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
        dX->numel(),
        scale_ / total_weight,
        dX->data<float>(),
        dX->template mutable_data<float>(),
        &context_);
  }
  math::Scale<float, float, CUDAContext>(
      dX->numel(),
      d_avg_loss.data<float>(),
      dX->data<float>(),
      dX->template mutable_data<float>(),
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

  Tensor* dX;
  if (only_loss_) {
    // Memory saving trick to share the buffer with the softmax output.
    // Softmax output is thus overwritten.
    dX = OutputTensorAlias(0, P);
    dX->ResizeLike(X);
  } else {
    dX = Output(0, X.sizes(), at::dtype<float>());
  }

  const auto canonical_axis = X.canonical_axis_index(1);
  int N, D;
  N = X.dim32(0);
  D = X.dim32(1);

  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));
  // Spatial mode, compute softmax for each x, y location
  CAFFE_ENFORCE_EQ(X.dim(), 4);
  CAFFE_ENFORCE_EQ(T.dim(), 3);

  int H = X.dim32(2);
  int W = X.dim32(3);
  dX->ResizeLike(X);
  if (!weights_.defined()) {
    weights_ = caffe2::empty({N * W * H}, at::dtype<float>().device(CUDA));
  } else if (weights_.numel() != N * W * H) {
    weights_.Resize(N * W * H);
  }

  const float* Pdata = P.data<float>();
  float* dX_data = dX->template mutable_data<float>();
  const int* label_data = T.data<int>();
  const float* d_avg_loss_data = d_avg_loss.data<float>();

  // Copy softmax probabilities into dX. All but the neuron
  // corresponding to the correct label has gradient equaling e(x_j)
  // which is the probability under softmax.
  context_.CopySameDevice<float>(P.numel(), Pdata, dX_data);

  math::Set<float, CUDAContext>(
      1, 0.0f, total_weight_ptr_.mutable_data<float>(), &context_);

  SpatialSoftmaxLossGradientKernel<<<
      CAFFE_GET_BLOCKS(N * W * H),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, W, H, label_data, weights, dX_data, weights_.mutable_data<float>());

  math::Sum<float, CUDAContext>(
      weights_.numel(),
      weights_.data<float>(),
      total_weight_ptr_.mutable_data<float>(),
      &context_,
      &scratch_);

  // Somewhat awkward scalar passing from device to host
  float h_total_weight;
  CUDA_CHECK(cudaMemcpyAsync(
      &h_total_weight,
      total_weight_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream()));

  // Final scaling
  if (h_total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
        dX->numel(),
        scale_ / h_total_weight,
        dX->data<float>(),
        dX->template mutable_data<float>(),
        &context_);
  }
  math::Scale<float, float, CUDAContext>(
      dX->numel(),
      d_avg_loss.data<float>(),
      dX->data<float>(),
      dX->template mutable_data<float>(),
      &context_);

  return true;
}

// Implementation for the CUDA context.
template <>
bool SoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  auto* P = Output(0, X.sizes(), at::dtype<float>());
  auto* P_data = P->mutable_data<float>();
  if (N == 0) {
    return true;
  }
  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }
  if (!scale_.defined()) {
    scale_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (scale_.numel() != N) {
    scale_.Resize(N);
  }

  if (!rowmax_.defined()) {
    rowmax_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (rowmax_.numel() != N) {
    rowmax_.Resize(N);
  }

  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      scale_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P_data,
      false,
      &context_);
  return true;
}
#define SOFTMAX_NUM_THREADS 128
#define RANDOM_SOFTMAX_NUM_THREADS 128

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

__global__ void random_softmax_gradient_kernel(
    const int rows,
    const int cols,
    const float* P_data,
    const float* label_data,
    const int* num_positive_labels,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(idx, rows * cols) {
    int i = idx / cols;
    dX[idx] = P_data[idx] * num_positive_labels[i] - label_data[idx];
  }
}
} // namespace

template <>
bool SoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  const auto canonical_axis = Y.canonical_axis_index(axis_);
  const int N = Y.size_to_dim(canonical_axis);
  const int D = Y.size_from_dim(canonical_axis);
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  auto* dX_data = dX->mutable_data<float>();
  if (N == 0) {
    return true;
  }
  softmax_gradient_kernel<<<
      N,
      SOFTMAX_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(D, Y.data<float>(), dY.data<float>(), dX_data);
  return true;
}

// Implementation for the CUDA context.
template <>
bool RandomSoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels/Targets
  CAFFE_ENFORCE(T.sizes() == X.sizes());

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  const int K = num_sampled_;

  auto* sampled_P = Output(
      0,
      vector<int64_t>{N, K},
      at::dtype<float>()); // sampled_softmax, shape (N, K)
  auto* sampled_T = Output(
      1,
      vector<int64_t>{N, K},
      at::dtype<float>()); // sampled_labels, shape (N, K)
  auto* avg_loss =
      Output(2, vector<int64_t>(), at::dtype<float>()); // Average loss, scalar
  auto* samples = Output(
      3,
      vector<int64_t>{N, K},
      at::dtype<int>()); // shape: (N, K), stored sampled classes. 0 <= elem < D
  auto* num_positive_labels =
      Output(4, vector<int64_t>{N}, at::dtype<int>()); // shape: N

  float* sampled_P_data = sampled_P->template mutable_data<float>();
  float* sampled_label_data = sampled_T->template mutable_data<float>();
  float* avg_loss_data = avg_loss->template mutable_data<float>();
  int* samples_data = samples->template mutable_data<int>();

  if (N == 0) {
    return true;
  }

  // Count positive labels per row
  RowwiseCountPositiveLabelsKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, T.data<float>(), num_positive_labels->mutable_data<int>());

  if (InputSize() >= 3) {
    auto& input_samples = Input(2);
    context_.CopySameDevice<int>(
        N * K, input_samples.data<int>(), samples_data);
  } else {
    if (!rand_.defined()) {
      rand_ = caffe2::empty({N * D}, at::dtype<float>().device(CUDA));
    } else if (rand_.numel() != N * D) {
      rand_.Resize(N * D);
    }

    float* rand_data = rand_.mutable_data<float>();
    CURAND_ENFORCE(
        curandGenerateUniform(context_.curand_generator(), rand_data, N * D));

    // sample classes:
    //   positive label classes per training example are always sampled
    //   sample K - k (k is num_positive_labels) negative label classes
    RandomSoftmaxSamplingKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        K,
        rand_data,
        T.data<float>(),
        num_positive_labels->data<int>(),
        samples_data);
  }

  // sample input
  SampleInputKernel<<<
      CAFFE_GET_BLOCKS(N * K),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, K, samples_data, X.data<float>(), sampled_P_data);
  SampleInputKernel<<<
      CAFFE_GET_BLOCKS(N * K),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, K, samples_data, T.data<float>(), sampled_label_data);

  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({K}, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(
        K, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != K) {
    sum_multiplier_.Resize(K);
    math::Set<float, CUDAContext>(
        K, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }
  if (!scale_.defined()) {
    scale_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (scale_.numel() != N) {
    scale_.Resize(N);
  }

  if (!rowmax_.defined()) {
    rowmax_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (rowmax_.numel() != N) {
    rowmax_.Resize(N);
  }

  Softmax(
      N,
      K,
      sampled_P->data<float>(),
      sum_multiplier_.data<float>(),
      scale_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      sampled_P_data,
      false,
      &context_);

  // Then compute cross entropy
  // Repurpose scale to store cross entropy loss per training example
  auto* scale = scale_.mutable_data<float>();
  CrossEntropyKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, K, sampled_P_data, sampled_label_data, scale);

  // Sum of all losses
  math::Sum<float, CUDAContext>(N, scale, avg_loss_data, &context_);
  // Average of input batch size
  math::Scale<float, float, CUDAContext>(
      1, 1.0 / N, avg_loss_data, avg_loss_data, &context_);

  return true;
}

template <>
bool RandomSoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& sampled_P = Input(0); // sampled_labels
  auto& sampled_T = Input(1); // sampled_softmax
  auto& samples = Input(2); // samples
  auto& d_avg_loss = Input(3); // gradient of avg_loss
  auto& T = Input(4); // labels
  auto& num_positive_labels = Input(5); // num positive labels per row
  auto* dX = Output(0, T.sizes(), at::dtype<float>());

  const auto canonical_axis = sampled_P.canonical_axis_index(axis_);
  const int N = sampled_P.size_to_dim(canonical_axis);
  const int K = sampled_P.size_from_dim(canonical_axis);
  const int D = T.size_from_dim(canonical_axis);

  float* dX_data = dX->mutable_data<float>();
  if (N == 0) {
    return true;
  }

  if (!sampled_dX_.defined()) {
    sampled_dX_ = caffe2::empty({N * K}, at::dtype<float>().device(CUDA));
  } else if (sampled_dX_.numel() != sampled_P.numel()) {
    sampled_dX_.Resize(N * K);
  }
  float* sampled_dX_data = sampled_dX_.mutable_data<float>();

  random_softmax_gradient_kernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      RANDOM_SOFTMAX_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      K,
      sampled_P.data<float>(),
      sampled_T.data<float>(),
      num_positive_labels.data<int>(),
      sampled_dX_data);

  math::Scale<float, float, CUDAContext>(
      sampled_dX_.size(),
      1.0f / N,
      sampled_dX_.data<float>(),
      sampled_dX_data,
      &context_);

  math::Scale<float, float, CUDAContext>(
      sampled_dX_.size(),
      d_avg_loss.data<float>(),
      sampled_dX_.data<float>(),
      sampled_dX_data,
      &context_);

  math::Set<float, CUDAContext>(T.size(), 0.0f, dX_data, &context_);

  SampleOutputKernel<<<
      CAFFE_GET_BLOCKS(N * K),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, K, samples.data<int>(), sampled_dX_data, dX_data);

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
REGISTER_CUDA_OPERATOR(RandomSoftmax, RandomSoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RandomSoftmaxGradient,
    RandomSoftmaxGradientOp<float, CUDAContext>);
} // namespace caffe2
