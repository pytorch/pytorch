#include <algorithm>

#include "caffe2/core/operator.h"
#include "caffe2/operators/segment_reduction_op.h"
#include "caffe2/operators/segment_reduction_op_gpu.cuh"
#include "caffe2/utils/GpuAtomics.cuh"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

void inclusive_scan_wrapper(
    const int* length_data,
    int len_length,
    Tensor* temp_buffer,
    Tensor* prefix_sum_out,
    CUDAContext* context_) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int)) / sizeof(int);
  temp_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(temp_buffer->template mutable_data<int>());
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
}

template <typename T, bool ExactBlock = false, bool Average = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void length_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post) {
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
    if (Average && (end - start) > 1) {
      sum /= (end - start);
    }
    out[group * post + threadIdx.x] = sum;
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T sum = (T)0;
      for (int line = start; line < end; ++line) {
        sum += in[line * post + i];
      }
      if (Average && (end - start) > 1) {
        sum /= (end - start);
      }
      out[group * post + i] = sum;
    }
  }
}

template <typename T, bool ExactBlock = false, bool Average = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void length_sum_gradient_kernel(
    const T* __restrict__ grad_in,
    T* __restrict__ grad_out,
    const int* __restrict__ prefix_sum_length_data,
    const int N,
    const int post) { //DESTROY
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
      if (Average && (end - start) > 1) {
        grad_out[line * post] /= (end - start);
      }
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      for (int line = start; line < end; ++line) {
        grad_out[line * post + i] = grad_in[group * post + i];
        if (Average && (end - start) > 1) {
          grad_out[line * post + i] /= (end - start);
        }
      }
    }
  }
}

template <typename T, bool ExactBlock = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void length_max_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    int N,
    int post,
    const T numeric_min) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    in += threadIdx.x;

    T max = numeric_min;
    for (int line = start; line < end; ++line) {
      T in_data = in[line * post];
      max = max > in_data ? max : in_data;
    }

    // setting output to 0 to not break gradient
    max = max == numeric_min ? 0 : max;
    out[group * post + threadIdx.x] = max;
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T max = numeric_min;
      for (int line = start; line < end; ++line) {
        T in_data = in[line * post + i];
        max = max > in_data ? max : in_data;
      }
      // setting output to 0 to not break gradient
      max = max == numeric_min ? 0 : max;
      out[group * post + i] = max;
    }
  }
}

template <typename T, bool ExactBlock = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void length_weighted_sum_gradient_kernel(
    const T* __restrict__ grad_in,
    const T *const __restrict__ weights_in,
    T* __restrict__ grad_out,
    const int *const __restrict__ prefix_sum_length_data,
    const int N,
    const int post) {
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
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void length_weighted_sum_with_main_input_gradient_kernel(
    const T *const __restrict__ grad_in,
    const T *const __restrict__ weights_in,
    const T *const __restrict__ data_in,
    const IndexType *const __restrict__ indices,
    T *const __restrict__ data_grad_out,
    T *const __restrict__ weights_grad_out,
    const int *const __restrict__ prefix_sum_length_data,
    const int N,
    const int post) {
  // len_length blocks
  const int group = blockIdx.x;

  const int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  const int end = prefix_sum_length_data[group];
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
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_length_max_kernel(
    const T* __restrict__ in,
    T *const __restrict__ out,
    const int *const __restrict__ prefix_sum_length_data,
    const IndexType *const __restrict__ indices,
    const int post,
    const int len_indices,
    const T numeric_min) {
  // len_length blocks
  const int group = blockIdx.x;

  const int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  const int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= len_indices);
  CUDA_KERNEL_ASSERT(end <= len_indices);

  extern __shared__ T reduceVals[];

  if (ExactBlock) {
    T max = numeric_min;

    in += threadIdx.x;
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      T in_data = in[indices[line] * post];
      max = max > in_data ? max : in_data;
    }

    reduceVals[threadIdx.y * blockDim.x + threadIdx.x] = max;
    __syncthreads();

    if (threadIdx.y == 0) {
      max = numeric_min;
      for (int i = 0; i < blockDim.y; ++i) {
        T in_data = reduceVals[i * blockDim.x + threadIdx.x];
        max = max > in_data ? max : in_data;
      }

      // setting output to 0 to not break gradient
      max = max == numeric_min ? 0 : max;
      out[group * post + threadIdx.x] = max;
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      T max = numeric_min;
      for (int line = start; line < end; ++line) {
        T in_data = in[indices[line] * post + i];
        max = max > in_data ? max : in_data;
      }
      // setting output to 0 to not break gradient
      max = max == numeric_min ? 0 : max;
      out[group * post + i] = max;
    }
  }
}

template <typename T, typename IndexType, bool ExactBlock = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_length_weighted_sum_kernel(
    const T* __restrict__ in,
    const T* __restrict__ in_weights,
    T* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    const IndexType* __restrict__ indices,
    int post,
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

  template <class... Args>
  explicit CUDASparseLengthsSumOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...) {}

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
    if (SparseFused) {
      return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(DATA));
    } else {
      return DoRunWithType2<IndexType, T>();
    }
  }

  template <typename IndexType, typename InType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const int64_t outputSize = lengthsInput.dim(0);
    const int len_length = outputSize;

    auto shape = dataInput.sizes().vec();
    shape[0] = outputSize;
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    const IndexType* indices;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
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

    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    if (SparseFused) {
      const InType* in_data = dataInput.template data<InType>();

      if (post <= maxThreads) {
        int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
        dim3 block(post, multiple);
        size_t smem = sizeof(T) * post * multiple;

        // calling cuda kernel with ExactBlock = true, Average = false
        sparse_length_sum_kernel<InType, T, IndexType, true, false>
            <<<len_length, block, smem, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        // calling cuda kernel with ExactBlock = false, Average = false
        sparse_length_sum_kernel<InType, T, IndexType, false, false>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      const T* in_data = dataInput.template data<T>();

      if (post <= maxThreads) {
        length_sum_kernel<T, true, false>
            <<<len_length, post, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        length_sum_kernel<T, true, false>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return true;
  }

  enum { DATA = 0, INDICES = 1, LENGTHS = 1 + (SparseFused ? 1 : 0) };

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
};

template <typename T, class Context = CUDAContext, bool SparseFused = true>
class CUDASparseLengthsMeanOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit CUDASparseLengthsMeanOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...) {}

  ~CUDASparseLengthsMeanOp() {}

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
    if (SparseFused) {
      return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(DATA));
    } else {
      return DoRunWithType2<IndexType, T>();
    }
  }

  template <typename IndexType, typename InType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const int64_t outputSize = lengthsInput.dim(0);
    const int len_length = outputSize;

    auto shape = dataInput.sizes().vec();
    shape[0] = outputSize;
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    const IndexType* indices;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
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

    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    if (SparseFused) {
      const InType* in_data = dataInput.template data<InType>();
      if (post <= maxThreads) {
        int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
        dim3 block(post, multiple);
        size_t smem = sizeof(T) * post * multiple;
        // calling cuda kernel with ExactBlock = true, Average = true
        sparse_length_sum_kernel<InType, T, IndexType, true, true>
            <<<len_length, block, smem, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        // calling cuda kernel with ExactBlock = false, Average = true
        sparse_length_sum_kernel<InType, T, IndexType, false, true>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                N,
                post,
                len_length,
                dataToReduceSize);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      const T* in_data = dataInput.template data<T>();

      if (post <= maxThreads) {
        // calling cuda kernel with ExactBlock = true, Average = true
        length_sum_kernel<T, true, true>
            <<<len_length, post, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        // calling cuda kernel with ExactBlock = true, Average = true
        length_sum_kernel<T, true, true>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data, out_data, prefix_sum_length_data, N, post);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return true;
  }

  enum { DATA = 0, INDICES = 1, LENGTHS = 1 + (SparseFused ? 1 : 0) };

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
};

template <typename T, class Context = CUDAContext, bool SparseFused = true>
class CUDASparseLengthsMaxOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit CUDASparseLengthsMaxOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...) {}

  ~CUDASparseLengthsMaxOp() {}

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

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const int64_t outputSize = lengthsInput.dim(0);
    int len_length = outputSize;

    auto shape = dataInput.sizes().vec();
    shape[0] = outputSize;
    auto* output = Output(0, shape, at::dtype<T>());

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    const IndexType* indices;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
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
    T* out_data = output->template mutable_data<T>();
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    T numeric_min = std::numeric_limits<T>::min();
    if (SparseFused) {
      if (post <= maxThreads) {
        const int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
        const dim3 block(post, multiple);
        const size_t smem = sizeof(T) * post * multiple;

        sparse_length_max_kernel<T, IndexType, true>
            <<<len_length, block, smem, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                post,
                dataToReduceSize,
                numeric_min);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        sparse_length_max_kernel<T, IndexType, false>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indices,
                post,
                dataToReduceSize,
                numeric_min);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      if (post <= maxThreads) {
        length_max_kernel<T, true>
            <<<len_length, post, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                N,
                post,
                numeric_min);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        length_max_kernel<T, true>
            <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                N,
                post,
                numeric_min);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return true;
  }

  enum { INDICES = 1, LENGTHS = 1 + (SparseFused ? 1 : 0) };

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
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

    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");
    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

    const int64_t dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    const int64_t dataToReduceSize = indicesInput.dim(0);
    const int64_t outputSize = lengthsInput.dim(0);
    const int len_length = outputSize;

    auto shape = dataInput.sizes().vec();
    shape[0] = outputSize;
    auto* output = Output(0, shape, at::dtype<T>());
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
    int post = dataInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    if (post <= maxThreads) {
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);
      size_t smem = sizeof(T) * post * multiple;

      sparse_length_weighted_sum_kernel<T, IndexType, true>
          <<<len_length, block, smem, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              indices,
              post,
              dataToReduceSize);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      sparse_length_weighted_sum_kernel<T, IndexType, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              indices,
              post,
              dataToReduceSize);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
  }

  enum { DATA = 0, WEIGHTS = 1, INDICES = 2, LENGTHS = 3 };

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
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
    gpu_atomic_add(&out[segment * slize_sz + j], data[i]);
    if (scales && j == 0) {
      gpu_atomic_add(&scales[segment], 1);
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

    if (segment_ids.numel() == 0 || data.numel() == 0) {
      // Special handling for empty input
      auto dims = data.sizes().vec();
      if (dims.size() > 0) {
        dims[0] = 0;
      }
      Output(0, dims, at::dtype<T>());
      return true;
    }

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t slize_sz = data.size_from_dim(1);

    ReinitializeTensor(&K_tensor_, {1}, at::dtype<SIndex>().device(CUDA));
    // Get maximum segment id so we can size the output.
    // This must be done synchronously with host.
    if (segment_ids.numel() > 4096) {
      // when the input size is large, device reduce is better.
      size_t tmp_storage_bytes = 0;
      // the first call to `Max` do nothing, but set correct tmp_storage_bytes.
      cub::DeviceReduce::Max(
          nullptr,
          tmp_storage_bytes,
          segment_ids.template data<SIndex>(), // input device data
          K_tensor_.template mutable_data<SIndex>(), // output device data
          segment_ids.numel(), // number of items
          context_.cuda_stream());

      // the second call do the real computation.
      ReinitializeTensor(
          &buffer_tensor_,
          {static_cast<int64_t>(tmp_storage_bytes)},
          at::dtype<char>().device(CUDA));
      cub::DeviceReduce::Max(
          static_cast<void*>(buffer_tensor_.mutable_data<char>()),
          tmp_storage_bytes,
          segment_ids.template data<SIndex>(), // input device data
          K_tensor_.template mutable_data<SIndex>(), // output device data
          segment_ids.numel(), // number of items
          context_.cuda_stream());
    } else {
      MaxSegmentKernel<SIndex>
          <<<1, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              segment_ids.numel(),
              segment_ids.template data<SIndex>(),
              K_tensor_.mutable_data<SIndex>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    SIndex K = 0;
    context_.CopyBytesToCPU(
        sizeof(SIndex), K_tensor_.template data<SIndex>(), &K);
    context_.FinishDeviceComputation();

    auto dims = data.sizes().vec();
    dims[0] = K + 1;
    auto* output = Output(0, dims, at::dtype<T>());

    // Clear the output as we will be accumulating the values
    math::Set<T, CUDAContext>(
        output->numel(), T(0), output->template mutable_data<T>(), &context_);

    if (!mean) {
      UnsortedSegmentSumKernel<SIndex, T>
          <<<CAFFE_GET_BLOCKS(data.numel()),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              data.numel(),
              slize_sz,
              segment_ids.template data<SIndex>(),
              data.template data<T>(),
              output->template mutable_data<T>(),
              nullptr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // For mean, we need to compute scaling factors
      ReinitializeTensor(
          &scaling_factors_, {K + 1}, at::dtype<int>().device(CUDA));
      math::Set<int, CUDAContext>(
          scaling_factors_.numel(),
          int(0),
          scaling_factors_.template mutable_data<int>(),
          &context_);
      UnsortedSegmentSumKernel<SIndex, T>
          <<<CAFFE_GET_BLOCKS(data.numel()),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              data.numel(),
              slize_sz,
              segment_ids.template data<SIndex>(),
              data.template data<T>(),
              output->template mutable_data<T>(),
              scaling_factors_.template mutable_data<int>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      // Divide by the scaling factors to get means
      SegmentScalingKernel<SIndex, T>
          <<<CAFFE_GET_BLOCKS(output->numel()),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              output->numel(),
              slize_sz,
              scaling_factors_.template data<int>(),
              output->template mutable_data<T>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
  }

 private:
  Tensor buffer_tensor_;
  Tensor K_tensor_;
  Tensor scaling_factors_; // for mean
};

template <typename SIndex>
__global__ void segment_lengths_kernel(int N, const SIndex* X, SIndex* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    gpu_atomic_add(&Y[X[i]], 1);
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
    auto dims = input.sizes().vec();
    SIndex K = 0;
    context_.CopyBytesToCPU(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
        <<<std::min(K, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            K,
            N,
            segment_len_prefix_sum_.template data<SIndex>(),
            segment_len_.template data<SIndex>(),
            input.template data<T>(),
            output->template mutable_data<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return true;
  }

 private:
  Tensor segment_len_{CUDA}; // for mean
  Tensor segment_len_prefix_sum_{CUDA};
  Tensor prefix_buffer_{CUDA};
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

    auto* dX = Output(0, X.sizes(), at::dtype<T>());

    const int M = X.dim32(0);
    const int N = X.size_from_dim(1);

    SIndex K = 0;
    context_.CopyBytesToCPU(
        sizeof(SIndex), I.template data<SIndex>() + I.numel() - 1, &K);

    K += 1;

    if (segment_len_.numel() != K) {
      ReinitializeTensor(&segment_len_, {K}, at::dtype<SIndex>().device(CUDA));
    }

    math::Set<SIndex, CUDAContext>(
        segment_len_.numel(),
        0,
        segment_len_.template mutable_data<SIndex>(),
        &context_);
    segment_lengths_kernel<<<
        CAFFE_GET_BLOCKS(I.numel()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        I.numel(),
        I.template data<SIndex>(),
        segment_len_.template mutable_data<SIndex>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sorted_segment_mean_gradient_kernel<T, SIndex, LOGEXP>
        <<<CAFFE_GET_BLOCKS(dX->numel()),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return true;
  }

 private:
  Tensor segment_len_; // for mean
};

REGISTER_CUDA_OPERATOR_STR(
    "LengthsSum",
    CUDASparseLengthsSumOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SparseLengthsSum",
    CUDASparseLengthsSumOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR_STR(
    "LengthsMean",
    CUDASparseLengthsMeanOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SparseLengthsMean",
    CUDASparseLengthsMeanOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR_STR(
    "LengthsMax",
    CUDASparseLengthsMaxOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR_STR(
    "SparseLengthsMax",
    CUDASparseLengthsMaxOp<float, CUDAContext, true>);
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

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.sizes().vec();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());
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
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);

      // calling cuda kernel with ExactBlock = true, Average = false
      length_sum_gradient_kernel<T, true, false>
          <<<len_length, block, 0, context_.cuda_stream()>>>(

              in_data, out_data, prefix_sum_length_data, N, post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // calling cuda kernel with ExactBlock = false, Average = false
      length_sum_gradient_kernel<T, false, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
};

template <typename T, class Context = CUDAContext>
class CUDASparseLengthsMeanGradientWithIndicesOp
    : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsMeanGradientWithIndicesOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsMeanGradientWithIndicesOp() {}

  bool RunOnDevice() override {
    auto& segmentGradsInput = Input(0);
    auto& lengthsInput = Input(1);
    auto& indicesInput = Input(2);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.sizes().vec();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());
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
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);

      // calling cuda kernel with ExactBlock = true, Average = true
      length_sum_gradient_kernel<T, true, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(

              in_data, out_data, prefix_sum_length_data, N, post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // calling cuda kernel with ExactBlock = false, Average = true
      length_sum_gradient_kernel<T, false, true>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
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

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.sizes().vec();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());
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
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);

      length_weighted_sum_gradient_kernel<T, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              N,
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      length_weighted_sum_gradient_kernel<T, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data,
              in_weights,
              out_data,
              prefix_sum_length_data,
              N,
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
};

template <typename T, bool ExactBlock = false>
__global__ void length_max_gradient_kernel(
    const T* __restrict__ grad_in,
    T* __restrict__ grad_out,
    const T* data_in,
    const T* data_out,
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
    data_in += threadIdx.x;
    data_out += threadIdx.x;

    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      if (data_in[line * post] == data_out[group * post]) {
        grad_out[line * post] = grad_in[group * post];
      } else {
        grad_out[line * post] = 0;
      }
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      for (int line = start; line < end; ++line) {
        if (data_in[line * post + i] == data_out[group * post + i]) {
          grad_out[line * post + i] = grad_in[group * post + i];
        } else {
          grad_out[line * post + i] = 0;
        }
      }
    }
  }
}

template <typename T, class Context = CUDAContext>
class CUDALengthsMaxWithMainInputAndForwardOutputGradientOp
    : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDALengthsMaxWithMainInputAndForwardOutputGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDALengthsMaxWithMainInputAndForwardOutputGradientOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, float>>::call(this, Input(3));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto& segmentGradsInput = Input(1);
    auto& lengthsInput = Input(2);
    auto& dataInput = Input(3);
    auto& dataOutput = Input(0); // based on CPU version

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

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

    auto shape = dataInput.sizes().vec();
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    const T* in_data = segmentGradsInput.template data<T>();
    T* out_data = dataGradsOutput->template mutable_data<T>();

    int N = dataInput.dim(0);
    int post = segmentGradsInput.size_from_dim(1);

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post <= maxThreads) {
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);

      length_max_gradient_kernel<T, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(

              in_data,
              out_data,
              dataInput.template data<T>(),
              dataOutput.template data<T>(),
              prefix_sum_length_data,
              N,
              post,
              len_length);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      length_max_gradient_kernel<T, false>
          <<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
              in_data,
              out_data,
              dataInput.template data<T>(),
              dataOutput.template data<T>(),
              prefix_sum_length_data,
              N,
              post,
              len_length);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
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

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    auto shape = segmentGradsInput.sizes().vec();
    int output_0dim = indicesInput.dim(0);
    shape[0] = output_0dim;
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());
    auto* weightGradsOutput = Output(1, indicesInput.sizes(), at::dtype<T>());
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
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
              post);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
};

// Needed because name is auto-generated in segment_reduction_op.cc:224
REGISTER_CUDA_OPERATOR_STR(
    "LengthsMaxWithMainInputAndForwardOutputGradient",
    CUDALengthsMaxWithMainInputAndForwardOutputGradientOp<float, CUDAContext>);

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

REGISTER_CUDA_OPERATOR(
    SparseLengthsIndicesInGradientMeanGradient,
    CUDASparseLengthsMeanGradientWithIndicesOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    LengthsIndicesInGradientMeanGradient,
    CUDASparseLengthsMeanGradientWithIndicesOp<float, CUDAContext>);
} // namespace caffe2

// Macro doesn't like comma
using LengthsSumCUDAOp =
    caffe2::CUDASparseLengthsSumOp<float, caffe2::CUDAContext, false>;
using LengthsMeanCUDAOp =
    caffe2::CUDASparseLengthsMeanOp<float, caffe2::CUDAContext, false>;
using LengthsMaxCUDAOp =
    caffe2::CUDASparseLengthsMaxOp<float, caffe2::CUDAContext, false>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(LengthsSum, LengthsSumCUDAOp);
C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(LengthsMean, LengthsMeanCUDAOp);
C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(LengthsMax, LengthsMaxCUDAOp);

#undef SEGREDUCE_MINBLOCKS
