#include <algorithm>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/boolean_mask_ops.h"
#include <cub/cub.cuh>
#include "caffe2/utils/cub_namespace.cuh"

namespace caffe2 {

namespace {
__global__ void BooleanMaskCopyKernel(
    const int64_t numOfOutput,
    const int64_t numBytes,
    const int64_t* indices,
    const uint8_t* src,
    uint8_t* dest) {
  for (int64_t i = blockIdx.x; i < numOfOutput; i += gridDim.x) {
    const auto srcBase = indices[i] * numBytes;
    const auto destBase = i * numBytes;
    for (int64_t j = threadIdx.x; j < numBytes; j += blockDim.x) {
      dest[destBase + j] = src[srcBase + j];
    }
  }
}
}

template <>
class BooleanMaskOp<CUDAContext> final : public Operator<CUDAContext> {
 public:
  BooleanMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& src = Input(0);
    const auto& mask = Input(1);
    auto* dest = Output(0);

    CAFFE_ENFORCE(src.dim() >= 1);
    CAFFE_ENFORCE_EQ(mask.dim(), 1);
    CAFFE_ENFORCE(src.size(0) == mask.size(0));

    const auto* maskData = mask.data<bool>();
    const auto outerSize = mask.size(0);
    ReinitializeTensor(
        &indices_, {outerSize}, at::dtype<int64_t>().device(CUDA));
    auto* indicesData = indices_.mutable_data<int64_t>();

    size_t numBytes = 0;
    cub::CountingInputIterator<int> itr(0);
    cub::DeviceSelect::Flagged(
        nullptr,
        numBytes,
        itr,
        maskData,
        indicesData,
        static_cast<int64_t*>(nullptr),
        outerSize,
        context_.cuda_stream());

    auto numint64_t =
        static_cast<int64_t>((numBytes + sizeof(int64_t) - 1) / sizeof(int64_t));
    // allocate one more int64_t at the end of scratch for storing numOfOutput
    ReinitializeTensor(
        &scratch_, {numint64_t + 1}, at::dtype<int64_t>().device(CUDA));
    auto* scratchData = scratch_.mutable_data<int64_t>();
    auto* numOfOutputData = scratchData + numint64_t;

    cub::DeviceSelect::Flagged(
        static_cast<void*>(scratchData),
        numBytes,
        itr,
        maskData,
        indicesData,
        numOfOutputData,
        outerSize,
        context_.cuda_stream());

    // Copy numOfOutput from gpu to cpu
    int64_t numOfOutput;
    context_.CopyToCPU(1, numOfOutputData, &numOfOutput);

    indices_.Resize(numOfOutput);
    std::vector<int64_t> dims = src.sizes().vec();
    dims[0] = numOfOutput;
    dest->Resize(dims);
    auto* destData = (uint8_t*)dest->raw_mutable_data(src.meta());
    const auto* srcData = (uint8_t*)src.raw_data();
    if (OutputSize() == 2) {

      auto* indicesOut = Output(1, {numOfOutput}, at::dtype<int64_t>());
      indicesOut->template mutable_data<int64_t>();
    }

    if (numOfOutput > 0) {
      BooleanMaskCopyKernel<<<
          std::min(numOfOutput, static_cast<int64_t>(CAFFE_MAXIMUM_NUM_BLOCKS)),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          numOfOutput,
          src.size_from_dim(1) * src.meta().itemsize(),
          indicesData,
          srcData,
          destData);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      if (OutputSize() == 2) {
        Output(1)->CopyFrom(indices_, /* async */ true);
      }
    }

    return true;
  }

 private:
  Tensor indices_;
  Tensor scratch_;
};

REGISTER_CUDA_OPERATOR(BooleanMask, BooleanMaskOp<CUDAContext>);

namespace {

#define minf (-1.0f * std::numeric_limits<float>::infinity())

template <typename T>
__global__ void sequenceMaskKernel(
    int N,
    int M,
    int B,
    const T* in,
    const int* seq_lengths,
    T fill_val,
    T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k >= seq_lengths[j] ? fill_val : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j >= seq_lengths[i] ? fill_val : in[index]);
    }
  }
}

template <typename T>
__global__ void repeatedSequenceMaskKernel(
    int N,
    int M,
    int D,
    const T* in,
    const int* seq_lengths,
    T fill_val,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, N * M * D) {
    int i = index / (D * M);
    int j = (index / D) % M;

    out[index] = (j >= seq_lengths[i] ? fill_val : in[index]);
  }
}

template <typename T>
__global__ void windowMaskKernel(
    int N,
    int M,
    int B,
    const T* in,
    const int* window_centers,
    const int radius,
    T fill_val,
    T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] =
          (k < window_centers[j] - radius || k > window_centers[j] + radius
               ? fill_val
               : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] =
          (j < window_centers[i] - radius || j > window_centers[i] + radius
               ? fill_val
               : in[index]);
    }
  }
}

template <typename T>
__global__ void
upperMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k > j ? fill_val : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j > i ? fill_val : in[index]);
    }
  }
}

template <typename T>
__global__ void
lowerMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k < j ? fill_val : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j < i ? fill_val : in[index]);
    }
  }
}

template <typename T>
__global__ void
upperDiagMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k >= j ? fill_val : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j >= i ? fill_val : in[index]);
    }
  }
}

template <typename T>
__global__ void
lowerDiagMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    CUDA_1D_KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k <= j ? fill_val : in[ind]);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j <= i ? fill_val : in[index]);
    }
  }
}

} // namespace

template <>
bool SequenceMaskOp<CUDAContext>::RunOnDevice() {
    return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
}

template <>
template <class T>
bool SequenceMaskOp<CUDAContext>::DoRunWithType() {
  const Tensor* input = &Input(0);
  const Tensor* sequence_lengths = nullptr;
  const Tensor* window_centers = nullptr;

  if (mode_ == "sequence") {
    sequence_lengths = &Input(1);
  } else if (mode_ == "window") {
    window_centers = &Input(1);
  }

  auto* output = Output(0, input->sizes(), at::dtype<T>());

  const auto canonical_axis = input->canonical_axis_index(axis_);

  // canonical_batch is non-negative if batching, -1 otherwise
  int canonical_batch = -1;
  if ((HasArgument("batch"))) {
    canonical_batch = input->canonical_axis_index(batch_);
  }

  // make sure batch < axis
  if (canonical_batch >= 0) {
    CAFFE_ENFORCE_LT(canonical_batch, canonical_axis);
  }

  // if no batch, then left is product of dims up to axis
  // otherwise, left is product of dims between batch and axis
  const int left =
      (canonical_batch >= 0
           ? input->size_between_dim(canonical_batch, canonical_axis)
           : input->size_to_dim(canonical_axis));
  const int right = input->size_from_dim(canonical_axis);

  // product of dims from 1 to batch
  const int batch_dim =
      (canonical_batch >= 0
           ? input->size_to_dim(canonical_batch) * input->dim(canonical_batch)
           : -1);

  T fill_val = convert::To<float, T>(grad_ ? 0.0f : fill_val_);
  if (mode_ == "sequence") {
    if (HasArgument("repeat_from_axis")) {
      const int canonical_repeat_from =
          input->canonical_axis_index(repeat_from_);
      const int repeated_dims = input->size_from_dim(canonical_repeat_from);
      const int masked_dims = right / repeated_dims;
      repeatedSequenceMaskKernel<<<
          CAFFE_GET_BLOCKS(left * right),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          left,
          masked_dims,
          repeated_dims,
          input->data<T>(),
          sequence_lengths->data<int>(),
          fill_val,
          output->template mutable_data<T>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      sequenceMaskKernel<<<
          CAFFE_GET_BLOCKS(left * right),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          left,
          right,
          batch_dim,
          input->data<T>(),
          sequence_lengths->data<int>(),
          fill_val,
          output->template mutable_data<T>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  } else if (mode_ == "window") {
    windowMaskKernel<<<
        CAFFE_GET_BLOCKS(left * right),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left,
        right,
        batch_dim,
        input->data<T>(),
        window_centers->data<int>(),
        radius_,
        fill_val,
        output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (mode_ == "upper") {
    upperMaskKernel<<<
        CAFFE_GET_BLOCKS(left * right),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left,
        right,
        batch_dim,
        input->data<T>(),
        fill_val,
        output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (mode_ == "lower") {
    lowerMaskKernel<<<
        CAFFE_GET_BLOCKS(left * right),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left,
        right,
        batch_dim,
        input->data<T>(),
        fill_val,
        output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (mode_ == "upperdiag") {
    upperDiagMaskKernel<<<
        CAFFE_GET_BLOCKS(left * right),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left,
        right,
        batch_dim,
        input->data<T>(),
        fill_val,
        output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (mode_ == "lowerdiag") {
    lowerDiagMaskKernel<<<
        CAFFE_GET_BLOCKS(left * right),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left,
        right,
        batch_dim,
        input->data<T>(),
        fill_val,
        output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    CAFFE_ENFORCE(false, "Unsupported mode for SequenceMaskOp!");
  }

  return true;
}

REGISTER_CUDA_OPERATOR(SequenceMask, SequenceMaskOp<CUDAContext>);

} // namespace caffe2
