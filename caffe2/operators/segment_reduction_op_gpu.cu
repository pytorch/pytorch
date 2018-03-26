#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"


namespace caffe2 {

namespace {

void inclusive_scan_wrapper(
    const int* length_data,
    int len_length,
    Tensor<CUDAContext>* temp_buffer,
    Tensor<CUDAContext>* prefix_sum_out,
    CUDAContext* context_) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->mutable_data<int>(),
      len_length,
      context_->cuda_stream());
  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int)) / sizeof(int);
  temp_buffer->Resize(buffer_size);
  void* d_temp_storage = static_cast<void*>(temp_buffer->mutable_data<int>());
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->mutable_data<int>(),
      len_length,
      context_->cuda_stream());
}

template <typename T, bool ExactBlock = false>
__global__ void length_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    in += threadIdx.x;

    T sum = (T)0;
    for (int line = start; line < end; ++line) {
      sum += in[line * post];
    }

    out[group * post + threadIdx.x] = sum;
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T sum = (T)0;
      for (int line = start; line < end; ++line) {
        sum += in[line * post + i];
      }
      out[group * post + i] = sum;
    }
  }
}

template <typename T, bool ExactBlock = false>
__global__ void length_sum_gradient_kernel(
    const T* __restrict__ grad_in,
    T* __restrict__ grad_out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    grad_out += threadIdx.x;
    grad_in += threadIdx.x;

    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      grad_out[line * post] = grad_in[group * post];
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      for (int line = start; line < end; ++line) {
        grad_out[line * post + i] = grad_in[group * post + i];
      }
    }
  }
}

template <typename T, bool ExactBlock = false>
__global__ void length_weighted_sum_gradient_kernel(
    const T* __restrict__ grad_in,
    const T* __restrict__ weights_in,
    T* __restrict__ grad_out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    grad_out += threadIdx.x;
    grad_in += threadIdx.x;

    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      grad_out[line * post] = weights_in[line] * grad_in[group * post];
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      for (int line = start; line < end; ++line) {
        grad_out[line * post + i] =
            weights_in[line] * grad_in[group * post + i];
      }
    }
  }
}

template <typename T, typename IndexType, int NumThreads>
__global__ void length_weighted_sum_with_main_input_gradient_kernel(
    const T* __restrict__ grad_in,
    const T* __restrict__ weights_in,
    const T* __restrict__ data_in,
    const IndexType* __restrict__ indices,
    T* __restrict__ data_grad_out,
    T* __restrict__ weights_grad_out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  // todo figure this num threads thing
  typedef cub::BlockReduce<float, NumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // TODO(wyiming): parallelize this outter loop
  for (int line = start; line < end; ++line) {
    T w_grad = 0;
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      auto g_in = grad_in[group * post + i];
      data_grad_out[line * post + i] = weights_in[line] * g_in;
      w_grad += g_in * data_in[indices[line] * post + i];
    }
    w_grad = BlockReduce(temp_storage).Reduce(w_grad, cub::Sum());
    if (threadIdx.x == 0) {
      weights_grad_out[line] = w_grad;
    }
    __syncthreads();
  }
}

template <typename T, typename IndexType, bool ExactBlock = false>
__global__ void sparse_length_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    const IndexType* __restrict__ indices,
    int N,
    int post,
    int len_length,
    int len_indices) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= len_indices);
  CUDA_KERNEL_ASSERT(end <= len_indices);

  extern __shared__ T reduceVals[];

  if (ExactBlock) {
    T sum = (T)0;

    in += threadIdx.x;
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      sum += in[indices[line] * post];
    }

    reduceVals[threadIdx.y * blockDim.x + threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
      sum = (T)0;
      for (int i = 0; i < blockDim.y; ++i) {
        sum += reduceVals[i * blockDim.x + threadIdx.x];
      }

      out[group * post + threadIdx.x] = sum;
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T sum = (T)0;
      for (int line = start; line < end; ++line) {
        sum += in[indices[line] * post + i];
      }
      out[group * post + i] = sum;
    }
  }
}

template <typename T, typename IndexType, bool ExactBlock = false>
__global__ void sparse_length_weighted_sum_kernel(
    const T* __restrict__ in,
    const T* __restrict__ in_weights,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    const IndexType* __restrict__ indices,
    int N,
    int post,
    int len_length,
    int len_indices) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= len_indices);
  CUDA_KERNEL_ASSERT(end <= len_indices);

  extern __shared__ T reduceVals[];

  if (ExactBlock) {
    T sum = (T)0;

    in += threadIdx.x;
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      sum += in_weights[line] * in[indices[line] * post];
    }

    reduceVals[threadIdx.y * blockDim.x + threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
      sum = (T)0;
      for (int i = 0; i < blockDim.y; ++i) {
        sum += reduceVals[i * blockDim.x + threadIdx.x];
      }

      out[group * post + threadIdx.x] = sum;
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T sum = (T)0;
      for (int line = start; line < end; ++line) {
        sum += in_weights[line] * in[indices[line] * post + i];
      }
      out[group * post + i] = sum;
    }
  }
}

} // namespace

template <typename T, class Context = CUDAContext, bool SparseFused = true>
class CUDASparseLengthsSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsSumOp() {}

  bool RunOnDevice() override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      // type doesn't matter
      return DoRunWithType<int32_t>();
    }
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto& dataInput = Input(0);
    auto& lengthsInput = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    const TIndex dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    TIndex dataToReduceSize;
    const TIndex outputSize = lengthsInput.dim(0);
    const int len_length = outputSize;

    auto shape = dataInput.dims();
    shape[0] = outputSize;
    output->Resize(shape);
    T* out_data = output->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    const IndexType* indices;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indicesInput.ndim(), "INDICES must be a vector");
      indices = indicesInput.template data<IndexType>();
      dataToReduceSize = indicesInput.dim(0);
    } else {
      dataToReduceSize = dataSize;
    }

    // only compute this the first time
    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    const T* in_data = dataInput.template data<T>();
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    if (SparseFused) {
      if (post <= maxThreads) {
        int multiple = std::min(maxThreads / post, 16);
        dim3 block(post, multiple);
        size_t smem = sizeof(T) * post * multiple;

        sparse_length_sum_kernel<T, IndexType, true>
            <<<len_length, block, smem, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
      } else {
        sparse_length_sum_kernel<T, IndexType, false>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
      }
    } else {
      if (post <= maxThreads) {
        length_sum_kernel<T, true>
            <<<len_length, post, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post, len_length);
      } else {
        length_sum_kernel<T, true>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post, len_length);
      }
    }
    return true;
  }

  enum { INDICES = 1, LENGTHS = 1 + (SparseFused ? 1 : 0) };

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

template <typename T, class Context = CUDAContext, bool SparseFused = true>
class CUDASparseLengthsWeightedSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsWeightedSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsWeightedSumOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto& dataInput = Input(DATA);
    auto& weightsInput = Input(WEIGHTS);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(1, weightsInput.ndim(), "WEIGHTS must be a vector");
    CAFFE_ENFORCE_EQ(1, indicesInput.ndim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");

    const TIndex dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    const TIndex dataToReduceSize = indicesInput.dim(0);
    const TIndex outputSize = lengthsInput.dim(0);
    const int len_length = outputSize;

    auto shape = dataInput.dims();
    shape[0] = outputSize;
    output->Resize(shape);
    T* out_data = output->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    const IndexType* indices = indicesInput.template data<IndexType>();
    const T* in_data = dataInput.template data<T>();
    const T* in_weights = weightsInput.template data<T>();
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    if (post <= maxThreads) {
      int multiple = std::min(maxThreads / post, 16);
      dim3 block(post, multiple);
      size_t smem = sizeof(T) * post * multiple;

      sparse_length_weighted_sum_kernel<T, IndexType, true>
          <<<len_length, block, smem, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              indices,
              N,
              post,
              len_length,
              dataToReduceSize);
    } else {
      sparse_length_weighted_sum_kernel<T, IndexType, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              indices,
              N,
              post,
              len_length,
              dataToReduceSize);
    }
    return true;
  }

  enum { DATA = 0, WEIGHTS = 1, INDICES = 2, LENGTHS = 3 };

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

template <typename SIndex>
__global__ void
MaxSegmentKernel(int n, const SIndex* segment_ids, SIndex* max_segment) {
  typedef cub::BlockReduce<SIndex, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int mx = 0;

  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    mx = segment_ids[j] > mx ? segment_ids[j] : mx;
  }
  SIndex max_seg = BlockReduce(temp_storage).Reduce(mx, cub::Max());
  if (threadIdx.x == 0) {
    *max_segment = max_seg;
  }
}

template <typename SIndex, typename T>
__global__ void UnsortedSegmentSumKernel(
    int n,
    int slize_sz,
    const SIndex* segments,
    const T* data,
    T* out,
    int* scales) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int slice_idx = i / slize_sz;
    int j = i % slize_sz;
    SIndex segment = segments[slice_idx];
    atomicAdd(&out[segment * slize_sz + j], data[i]);
    if (scales && j == 0) {
      atomicAdd(&scales[segment], 1);
    }
  }
}

template <typename SIndex, typename T>
__global__ void
SegmentScalingKernel(int m, int slize_sz, const int* scales, T* out) {
  CUDA_1D_KERNEL_LOOP(i, m) {
    int scale = scales[i / slize_sz];
    out[i] = scale > 0 ? out[i] / scale : 0.0; // avoid 0/0 division
  }
}

template <typename T, typename SIndex, bool mean>
class CUDAUnsortedSegmentSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CUDAUnsortedSegmentSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDAUnsortedSegmentSumOp() {}

  bool RunOnDevice() override {
    auto& data = Input(0);
    auto& segment_ids = Input(1);
    auto* output = Output(0);

    if (segment_ids.size() == 0 || data.size() == 0) {
      // Special handling for empty input
      auto dims = data.dims();
      if (dims.size() > 0) {
        dims[0] = 0;
      }
      output->Resize(dims);
      output->template mutable_data<T>();
      return true;
    }

    CAFFE_ENFORCE_EQ(1, segment_ids.ndim(), "SEGMENT_IDS must be a vector");
    TIndex slize_sz = data.size_from_dim(1);

    K_tensor_.Resize(1);
    // Get maximum segment id so we can size the output.
    // This must be done synchronously with host.
    if (segment_ids.size() > 4096) {
      // when the input size is large, device reduce is better.
      size_t tmp_storage_bytes = 0;
      // the first call to `Max` do nothing, but set correct tmp_storage_bytes.
      cub::DeviceReduce::Max(
          nullptr,
          tmp_storage_bytes,
          segment_ids.template data<SIndex>(), // input device data
          K_tensor_.template mutable_data<SIndex>(), // output device data
          segment_ids.size(), // number of items
          context_.cuda_stream());

      // the second call do the real computation.
      buffer_tensor_.Resize(tmp_storage_bytes);
      cub::DeviceReduce::Max(
          static_cast<void*>(buffer_tensor_.mutable_data<char>()),
          tmp_storage_bytes,
          segment_ids.template data<SIndex>(), // input device data
          K_tensor_.template mutable_data<SIndex>(), // output device data
          segment_ids.size(), // number of items
          context_.cuda_stream());
    } else {
      MaxSegmentKernel<SIndex>
          <<<1, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              segment_ids.size(),
              segment_ids.template data<SIndex>(),
              K_tensor_.mutable_data<SIndex>());
    }

    SIndex K = 0;
    context_.CopyBytes<CUDAContext, CPUContext>(
        sizeof(SIndex), K_tensor_.template data<SIndex>(), &K);
    context_.FinishDeviceComputation();

    auto dims = data.dims();
    dims[0] = K + 1;
    output->Resize(dims);

    // Clear the output as we will be accumulating the values
    math::Set<T, CUDAContext>(
        output->size(), T(0), output->template mutable_data<T>(), &context_);

    if (!mean) {
      UnsortedSegmentSumKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(data.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          data.size(),
          slize_sz,
          segment_ids.template data<SIndex>(),
          data.template data<T>(),
          output->template mutable_data<T>(),
          nullptr);
    } else {
      // For mean, we need to compute scaling factors
      scaling_factors_.Resize(K + 1);
      math::Set<int, CUDAContext>(
          scaling_factors_.size(),
          int(0),
          scaling_factors_.template mutable_data<int>(),
          &context_);
      UnsortedSegmentSumKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(data.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          data.size(),
          slize_sz,
          segment_ids.template data<SIndex>(),
          data.template data<T>(),
          output->template mutable_data<T>(),
          scaling_factors_.template mutable_data<int>());
      // Divide by the scaling factors to get means
      SegmentScalingKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(output->size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output->size(),
          slize_sz,
          scaling_factors_.template data<int>(),
          output->template mutable_data<T>());
    }
    return true;
  }

 private:
  Tensor<CUDAContext> buffer_tensor_;
  Tensor<CUDAContext> K_tensor_;
  Tensor<CUDAContext> scaling_factors_; // for mean
};

template <typename SIndex>
__global__ void segment_lengths_kernel(int N, const SIndex* X, SIndex* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    atomicAdd(&Y[X[i]], 1);
  }
}

template <typename T, typename SIndex, bool LOGEXP = false>
__global__ void sorted_segment_mean_kernel(
    const SIndex K,
    const int N,
    const SIndex* S,
    const SIndex* I,
    const T* X,
    T* Y) {
  for (int sId = blockIdx.x; sId < K; sId += gridDim.x) {
    const int start_index = sId > 0 ? S[sId] * N : 0;
    const int y_start_index = sId * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      T sum = 0.0;
      for (int j = 0; j < I[sId]; ++j) {
        const T x_i_j = X[start_index + j * N + i];
        sum += LOGEXP ? exp(x_i_j) : x_i_j;
      }
      const T norm_sum = sum / I[sId];
      Y[y_start_index + i] = LOGEXP ? log(norm_sum) : norm_sum;
    }
  }
}

template <typename T, typename SIndex, bool LOGEXP, class Context = CUDAContext>
class SortedSegmentRangeMeanOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SortedSegmentRangeMeanOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  ~SortedSegmentRangeMeanOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(0);
    const auto& indices = Input(1);
    int M = input.dim32(0);
    int N = input.size_from_dim(1);
    auto* output = Output(0);
    auto dims = input.dims();
    SIndex K = 0;
    context_.template CopyBytes<Context, CPUContext>(
        sizeof(SIndex),
        indices.template data<SIndex>() + indices.size() - 1,
        &K);
    context_.FinishDeviceComputation();
    K += 1;
    dims[0] = K;
    if (segment_len_.size() != K) {
      segment_len_.Resize(K);
      segment_len_prefix_sum_.Resize(K);
    }
    output->Resize(dims);
    math::Set<SIndex, CUDAContext>(
        segment_len_.size(),
        0,
        segment_len_.template mutable_data<SIndex>(),
        &context_);
    segment_lengths_kernel<<<
        CAFFE_GET_BLOCKS(indices.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        indices.size(),
        indices.template data<SIndex>(),
        segment_len_.template mutable_data<SIndex>());
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        segment_len_.template data<SIndex>(),
        segment_len_prefix_sum_.template mutable_data<SIndex>(),
        K,
        context_.cuda_stream());
    auto buffer_size = (temp_storage_bytes + sizeof(T)) / sizeof(T);
    prefix_buffer_.Resize(buffer_size);
    void* dev_temp_storage =
        static_cast<void*>(prefix_buffer_.mutable_data<T>());
    cub::DeviceScan::ExclusiveSum(
        dev_temp_storage,
        temp_storage_bytes,
        segment_len_.template data<SIndex>(),
        segment_len_prefix_sum_.template mutable_data<SIndex>(),
        K,
        context_.cuda_stream());
    sorted_segment_mean_kernel<T, SIndex, LOGEXP>
        <<<min(K, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            K,
            N,
            segment_len_prefix_sum_.template data<SIndex>(),
            segment_len_.template data<SIndex>(),
            input.template data<T>(),
            output->template mutable_data<T>());
    return true;
  }

 private:
  Tensor<CUDAContext> segment_len_; // for mean
  Tensor<CUDAContext> segment_len_prefix_sum_;
  Tensor<CUDAContext> prefix_buffer_;
};

template <typename T, typename SIndex, bool LOGEXP = false>
__global__ void sorted_segment_mean_gradient_kernel(
    const int M,
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
    const SIndex* I,
    const SIndex* S,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    const int sId = I[i / N];
    const int sSize = S[sId];
    const int yId = N * sId + i % N;
    dX[i] = LOGEXP ? dY[yId] * exp(X[i] - Y[yId]) / sSize : dY[yId] / sSize;
  }
}

template <typename T, typename SIndex, bool LOGEXP, class Context = CUDAContext>
class SortedSegmentRangeMeanGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SortedSegmentRangeMeanGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}
  ~SortedSegmentRangeMeanGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    const auto& dY = Input(2);
    const auto& I = Input(3);
    auto* dX = Output(0);
    dX->ResizeLike(X);

    const int M = X.dim32(0);
    const int N = X.size_from_dim(1);

    SIndex K = 0;
    context_.template CopyBytes<Context, CPUContext>(
        sizeof(SIndex), I.template data<SIndex>() + I.size() - 1, &K);

    K += 1;

    if (segment_len_.size() != K) {
      segment_len_.Resize(K);
    }

    math::Set<SIndex, CUDAContext>(
        segment_len_.size(),
        0,
        segment_len_.template mutable_data<SIndex>(),
        &context_);
    segment_lengths_kernel<<<
        CAFFE_GET_BLOCKS(I.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        I.size(),
        I.template data<SIndex>(),
        segment_len_.template mutable_data<SIndex>());
    sorted_segment_mean_gradient_kernel<T, SIndex, LOGEXP>
        <<<CAFFE_GET_BLOCKS(dX->size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            M,
            N,
            X.template data<T>(),
            Y.template data<T>(),
            dY.template data<T>(),
            I.template data<SIndex>(),
            segment_len_.template data<SIndex>(),
            dX->template mutable_data<T>());

    return true;
  }

 private:
  Tensor<CUDAContext> segment_len_; // for mean
};

REGISTER_CUDA_OPERATOR_STR(
    "LengthsSum",
    CUDASparseLengthsSumOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SparseLengthsSum",
    CUDASparseLengthsSumOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR_STR(
    "SparseLengthsWeightedSum",
    CUDASparseLengthsWeightedSumOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR_STR(
    "UnsortedSegmentSum",
    CUDAUnsortedSegmentSumOp<float, int, false>);
REGISTER_CUDA_OPERATOR_STR(
    "UnsortedSegmentMean",
    CUDAUnsortedSegmentSumOp<float, int, true>);
REGISTER_CUDA_OPERATOR_STR(
    "SortedSegmentRangeMean",
    SortedSegmentRangeMeanOp<float, int, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SortedSegmentRangeLogMeanExp",
    SortedSegmentRangeMeanOp<float, int, true>);
REGISTER_CUDA_OPERATOR_STR(
    "SortedSegmentRangeMeanGradient",
    SortedSegmentRangeMeanGradientOp<float, int, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SortedSegmentRangeLogMeanExpGradient",
    SortedSegmentRangeMeanGradientOp<float, int, true>);

template <typename T, class Context = CUDAContext>
class CUDASparseLengthsSumGradientWithIndicesOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsSumGradientWithIndicesOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsSumGradientWithIndicesOp() {}

  bool RunOnDevice() override {
    auto& segmentGradsInput = Input(0);
    auto& lengthsInput = Input(1);
    auto& indicesInput = Input(2);
    auto* dataGradsOutput = Output(0);
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.ndim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.dims();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    dataGradsOutput->Resize(shape);
    T* out_data = dataGradsOutput->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const T* in_data = segmentGradsInput.template data<T>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post <= maxThreads) {
      int multiple = std::min(maxThreads / post, 16);
      dim3 block(post, multiple);

      length_sum_gradient_kernel<T, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(

              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    } else {
      length_sum_gradient_kernel<T, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    }

    return true;
  }

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

template <typename T, class Context = CUDAContext>
class CUDASparseLengthsWeightedSumGradientWithIndicesOp
    : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsWeightedSumGradientWithIndicesOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsWeightedSumGradientWithIndicesOp() {}

  bool RunOnDevice() override {
    auto& weightsInput = Input(0);
    auto& segmentGradsInput = Input(1);
    auto& lengthsInput = Input(2);
    auto& indicesInput = Input(3);
    auto* dataGradsOutput = Output(0);
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.ndim(), "WEIGHTS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.ndim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.dims();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    dataGradsOutput->Resize(shape);
    T* out_data = dataGradsOutput->template mutable_data<T>();
    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const T* in_data = segmentGradsInput.template data<T>();
    const T* in_weights = weightsInput.template data<T>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);
    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post < maxThreads) {
      int multiple = std::min(maxThreads / post, 16);
      dim3 block(post, multiple);

      length_weighted_sum_gradient_kernel<T, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    } else {
      length_weighted_sum_gradient_kernel<T, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    }

    return true;
  }

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

template <typename T, class Context = CUDAContext>
class CUDASparseLengthsIndicesInGradientWeightedSumWithMainInputGradientOp
    : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsIndicesInGradientWeightedSumWithMainInputGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsIndicesInGradientWeightedSumWithMainInputGradientOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(4));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto& weightsInput = Input(0);
    auto& segmentGradsInput = Input(1);
    auto& lengthsInput = Input(2);
    auto& dataInput = Input(3);
    auto& indicesInput = Input(4);
    auto* dataGradsOutput = Output(0);
    auto* weightGradsOutput = Output(1);
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.ndim(), "WEIGHTS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.ndim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.dims();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    dataGradsOutput->Resize(shape);
    weightGradsOutput->ResizeLike(indicesInput);
    T* out_data_grads = dataGradsOutput->template mutable_data<T>();
    T* out_weight_grads = weightGradsOutput->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const T* in_data = dataInput.template data<T>();
    const T* in_grads = segmentGradsInput.template data<T>();
    const T* in_weights = weightsInput.template data<T>();
    const IndexType* indices = indicesInput.template data<IndexType>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    if (post > 128) {
      length_weighted_sum_with_main_input_gradient_kernel<T, IndexType, 512>
          <<<len_length, 512, 0, context_.cuda_stream()>>>(
              in_grads,
              in_weights,
              in_data,
              indices,
              out_data_grads,
              out_weight_grads,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    } else if (post > 64) {
      length_weighted_sum_with_main_input_gradient_kernel<T, IndexType, 128>
          <<<len_length, 128, 0, context_.cuda_stream()>>>(
              in_grads,
              in_weights,
              in_data,
              indices,
              out_data_grads,
              out_weight_grads,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    } else if (post > 32) {
      length_weighted_sum_with_main_input_gradient_kernel<T, IndexType, 64>
          <<<len_length, 64, 0, context_.cuda_stream()>>>(
              in_grads,
              in_weights,
              in_data,
              indices,
              out_data_grads,
              out_weight_grads,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    } else {
      length_weighted_sum_with_main_input_gradient_kernel<T, IndexType, 32>
          <<<len_length, 32, 0, context_.cuda_stream()>>>(
              in_grads,
              in_weights,
              in_data,
              indices,
              out_data_grads,
              out_weight_grads,
              prefix_sum_length_data,
              N,
              post,
              len_length);
    }

    return true;
  }

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

REGISTER_CUDA_OPERATOR(
    SparseLengthsIndicesInGradientWeightedSumGradient,
    CUDASparseLengthsWeightedSumGradientWithIndicesOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    CUDASparseLengthsIndicesInGradientWeightedSumWithMainInputGradientOp<
        float,
        CUDAContext>);

REGISTER_CUDA_OPERATOR(
    SparseLengthsIndicesInGradientSumGradient,
    CUDASparseLengthsSumGradientWithIndicesOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    LengthsIndicesInGradientSumGradient,
    CUDASparseLengthsSumGradientWithIndicesOp<float, CUDAContext>);
} // namespace caffe2
