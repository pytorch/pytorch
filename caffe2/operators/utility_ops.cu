#include <math.h>
#include <cfloat>
// TODO(jamesreed): I would use <cmath> here but std::isnan
// and std::isinf are declared constexpr there and the nvidia
// compiler throws an error because of it

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>
#include "caffe2/core/context_gpu.h"
#include "flatten_op.h"
#include "minmax_ops.h"
#include "utility_ops.h"

namespace caffe2 {
CAFFE_KNOWN_TYPE(const float*);

REGISTER_CUDA_OPERATOR(EnsureDense, EnsureDenseOp<CUDAContext>);

__global__ void NanCheckKernel(int N, const float* X, bool* result) {
  bool has_nan = false;
  CUDA_1D_KERNEL_LOOP(i, N) {
    // Note: we have no need to do early return, since only if this fails
    // will we not need to inspect all elements. No need to optimize the
    // case that will fail.
    has_nan = has_nan || isnan(X[i]) || isinf(X[i]);
  }
  __syncthreads();
  if (has_nan) {
    result[0] = true;
  }
}

template <>
bool NanCheckOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const size_t N = X.size();
  const float* data_ptr = X.data<float>();

  scratch_.Resize(1);
  math::Set<bool, CUDAContext>(
      1, false, scratch_.mutable_data<bool>(), &context_);
  NanCheckKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, X.data<float>(), scratch_.mutable_data<bool>());

  bool result = false;
  {
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    CUDA_ENFORCE(cudaMemcpyAsync(
        &result,
        scratch_.raw_data(),
        1,
        cudaMemcpyDefault,
        context_.cuda_stream()));
  }
  // Note: we must synchronize here so we can inspect the result
  context_.FinishDeviceComputation();

  // Print out diagnostic info if we have a NaN or inf
  if (result) {
    std::cerr << "Tensor contained NaN or inf: " << this->debug_def().input(0)
              << std::endl;

    for (int j = 0; j < InputSize(); j++) {
      TensorCPU cpu_X;
      cpu_X.ResizeLike(Input(j));
      // Hack to cause allocaiton happen here, so it won't happen
      // when we do CopyFrom. We need the mutex then because host->gpu
      // copies seem to possibly lock with NCCL.
      cpu_X.mutable_data<float>();

      {
        std::lock_guard<std::mutex> lock(CUDAContext::mutex());
        cpu_X.CopyFrom(Input(j), &context_);
      }
      context_.FinishDeviceComputation();
      std::cerr << "Input tensor: " << j << ": [" << this->debug_def().input(j)
                << "]" << std::endl;
      tensorPrinter_.Print<float>(cpu_X);

      if (j == 0) {
        std::cerr << "NaN idxs:" << std::endl;
        auto* cpu_X_data = cpu_X.data<float>();
        for (size_t i = 0; i < cpu_X.size(); ++i) {
          if (isnan(cpu_X_data[i]) || isinf(cpu_X_data[i])) {
            std::cerr << i << " ";
          }
        }
      }
      std::cerr << std::endl;
    }
    return false;
  }

  // This op should act as an identity matrix if we don't find any NaNs/infs.
  // Copy over the data if we are not doing this in-place.
  if (&X != Y) {
    Y->CopyFrom(X, &context_);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(NanCheck, NanCheckOp<CUDAContext>);

__global__ void
ElwiseMaxKernel(const float* X, const float* Y, float* maxout, const int N) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    maxout[i] = max(X[i], Y[i]);
  }
}

template <>
bool MaxOp<float, CUDAContext>::Compute() {
  float* output_data = Output(0)->mutable_data<float>();
  const int N = Input(0).size();

  // Run pairwise-maxes
  for (int i = 1; i < InputSize(); ++i) {
    ElwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        (i == 0 ? Input(0).data<float>() : Output(0)->data<float>()),
        Input(i).data<float>(),
        output_data,
        N);
  }

  return true;
}

REGISTER_CUDA_OPERATOR(Max, MaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MaxGradient, MaxGradientOp<float, CUDAContext>);

__global__ void
ElwiseMinKernel(const float* X, const float* Y, float* minout, const int N) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    minout[i] = min(X[i], Y[i]);
  }
}

template <>
bool MinOp<float, CUDAContext>::Compute() {
  float* output_data = Output(0)->mutable_data<float>();
  const int N = Input(0).size();

  // Run pairwise-mines
  for (int i = 1; i < InputSize(); ++i) {
    ElwiseMinKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        (i == 0 ? Input(0).data<float>() : Output(0)->data<float>()),
        Input(i).data<float>(),
        output_data,
        N);
  }

  return true;
}

REGISTER_CUDA_OPERATOR(Min, MinOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MinGradient, MinGradientOp<float, CUDAContext>);

template <typename T>
__global__ void
MaxMinGradKernel(int N, const T* mx, const T* x, const T* go, T* gi) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    gi[i] = go[i] * (mx[i] == x[i]);
  }
}

template <>
bool SelectGradientOpBase<float, CUDAContext>::RunOnDevice() {
  auto& output = Input(0);
  auto& grad_output = Input(1);
  const int kInputStartOffset = 2;

  const float* data = output.data<float>();

  for (int i = 0; i < OutputSize(); i++) {
    auto& input = Input(i + kInputStartOffset);
    auto* grad_input = Output(i);
    grad_input->ResizeLike(input);
    MaxMinGradKernel<<<
        CAFFE_GET_BLOCKS(input.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        input.size(),
        output.data<float>(),
        input.data<float>(),
        grad_output.data<float>(),
        grad_input->mutable_data<float>());
  }
  return true;
}

template <typename T_INDEX>
__global__ void GatherKernel(
    const float* X,
    float* Y,
    const T_INDEX* indices,
    const int N,
    const int block_size) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = indices[i];
    const float* src_offset = X + idx * block_size;
    float* dst_offset = Y + i * block_size;
    for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
      dst_offset[j] = src_offset[j];
    }
  }
}

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename Index>
bool GatherOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  auto shape = indices.dims();
  shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
  output->Resize(shape);

  int block_size = data.size() / data.dim(0);
  auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
  CAFFE_ENFORCE(
      block_bytesize == data.nbytes() / data.dim(0),
      "block_bytesize should be consistent with data dim");
  int N = indices.size();

  auto src_base = static_cast<const float*>(data.raw_data());
  const Index* idxs = indices.template data<Index>();
  auto out = static_cast<float*>(output->raw_mutable_data(data.meta()));

  // return early when the input is empty, since CUDA kernel will fail for
  // empty input.
  if (N <= 0) {
    return true;
  }

  GatherKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(src_base, out, idxs, N, block_size);
  return true;
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);

/**
 * @brief Update slices of Y in-place with a batch of weighted X's.
 * Y[idx] = alpha[b] * X[b][i] + Y[idx]
 * i=0,...,N-1
 * b=0,...,B-1
 * idx=Indices[i]
 */
template <typename T_INDEX>
__global__ void AxpySliceKernel(
    const float* weight0,
    const TIndex N,
    const TIndex B,
    const TIndex slice_size,
    const float** alpha,
    const float** X,
    const T_INDEX* Indices,
    float* Y,
    const TIndex M) {
  // This implementation requires that the first weight is 1.0
  CUDA_KERNEL_ASSERT(weight0[0] == 1.0);
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = Indices[i];
    float* y_offset = Y + (idx * slice_size);
    for (int b = 0; b < B; b++) {
      float a = *alpha[b];
      const float* x_offset = X[b] + (i * slice_size);
      for (int j = threadIdx.x; j < slice_size; j += blockDim.x) {
        atomicAdd(&y_offset[j], a * x_offset[j]);
      }
    }
  }
}

template <>
bool ScatterWeightedSumOp<float, CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
}

template <>
template <typename Index>
bool ScatterWeightedSumOp<float, CUDAContext>::DoRunWithType() {
  CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
  auto& X0 = Input(0);
  auto& weight0 = Input(1);
  auto& indices = Input(2);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");
  CAFFE_ENFORCE_GT(X0.size(), 0);
  CAFFE_ENFORCE_GT(X0.ndim(), 0, "X0 has to be at least the vector");
  CAFFE_ENFORCE_EQ(weight0.size(), 1);

  TIndex M = X0.size();
  TIndex N = X0.dim(0);
  TIndex K = indices.size();
  TIndex block_size = M / N;

  T* data = output->template mutable_data<T>();

  // In order to have all device pointers of x_i (and weight_i similarly)
  // consecutively in device memory, copy pointers to a host vector and then
  // copy back into a device array.
  const TIndex B = (InputSize() - 3) / 2;
  x_data_host_.Resize(B);
  weights_host_.Resize(B);
  x_data_device_.Resize(B);
  weights_device_.Resize(B);

  const float** x_data_host = x_data_host_.mutable_data<const float*>();
  const float** weights_host = weights_host_.mutable_data<const float*>();
  const float** x_data_device = x_data_device_.mutable_data<const float*>();
  const float** weights_device = weights_device_.mutable_data<const float*>();

  for (int inp = 3; inp < InputSize(); inp += 2) {
    int idx = (inp - 3) / 2;
    x_data_host[idx] = static_cast<const float*>(Input(inp).raw_data());
    weights_host[idx] = static_cast<const float*>(Input(inp + 1).raw_data());
  }
  context_.Copy<const float*, CPUContext, CUDAContext>(
      B, x_data_host, x_data_device);
  context_.Copy<const float*, CPUContext, CUDAContext>(
      B, weights_host, weights_device);

  AxpySliceKernel<<<
      std::min<TIndex>(K, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      weight0.template data<float>(),
      K,
      B,
      block_size,
      weights_device,
      x_data_device,
      indices.template data<Index>(),
      data,
      M);

  return true;
}

REGISTER_CUDA_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CUDAContext>);

namespace {

template <typename Index, typename T>
__global__ void scatter_assign_kernel(
    T* data,
    const Index* idxs,
    const T* slicesData,
    TIndex N,
    TIndex K,
    TIndex block_size) {
  for (TIndex i = blockIdx.x; i < K; i += gridDim.x) {
    Index idx = idxs[i];
    CUDA_KERNEL_ASSERT(0 <= idx && idx < N);
    const T* src = slicesData + block_size * i;
    T* dest = data + block_size * idx;
    for (TIndex j = threadIdx.x; j < block_size; j += blockDim.x) {
      dest[j] = src[j];
    }
  }
}

} // namespace

template <>
template <typename Index, typename T>
void ScatterAssignOp<CUDAContext>::DoScatterAssign(
    T* data,
    const Index* idxs,
    const T* slicesData,
    TIndex N,
    TIndex K,
    TIndex block_size) {
  scatter_assign_kernel<<<
      std::min(K, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(data, idxs, slicesData, N, K, block_size);
}

REGISTER_CUDA_OPERATOR(ScatterAssign, ScatterAssignOp<CUDAContext>);

#if THRUST_VERSION >= 100800
__global__ void remap_kernel(
    thrust::device_ptr<int> second_order,
    thrust::device_ptr<int> order,
    int* output,
    int N,
    int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= K)
    return;
  int idx = second_order[i];
  output[order[idx]] = i;
  // Maybe cuda 1D kernel?
  for (idx++; idx < N && (i == K - 1 || idx != second_order[i + 1]); idx++) {
    output[order[idx]] = i;
  }
  return;
}

template <>
template <typename T>
void UniqueOp<CUDAContext>::DoRun() {
  auto& inputTensor = Input(0);
  // use dim32 to enforce that it's fine to have remapping of type int
  int N = inputTensor.dim32(0);
  CAFFE_ENFORCE_EQ(inputTensor.ndim(), 1, "Input should be a vector");
  auto* uniqueTensor = Output(UNIQUE);

  int* remapping = nullptr;
  if (REMAPPING < OutputSize()) {
    auto* remappingTensor = Output(REMAPPING);
    remappingTensor->ResizeLike(inputTensor);
    remapping = remappingTensor->template mutable_data<int>();
  }

  const T* input = inputTensor.template data<T>();
  thrust_unique_buffer_.Resize(N);
  auto* buffer = thrust_unique_buffer_.template mutable_data<T>();
  context_.template CopyItems<CUDAContext, CUDAContext>(
      inputTensor.meta(), N, input, buffer);

  // Create two vectors of {0, 1, ..., N-1} on CUDA device
  thrust::device_vector<int> order1(N), order2(N);
  thrust::sequence(
      thrust::cuda::par.on(context_.cuda_stream()),
      order1.begin(),
      order1.end());
  thrust::sequence(
      thrust::cuda::par.on(context_.cuda_stream()),
      order2.begin(),
      order2.end());

  // Sort the input along with order vector. So now we know where each element
  // is permutated to. For example:
  //    input1 = 1,3,5,1,5,7,9
  //    order1 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,1,3,5,5,7,9
  //    order1 = 0,3,1,2,4,5,6
  thrust::sort_by_key(
      thrust::cuda::par.on(context_.cuda_stream()),
      buffer,
      buffer + N,
      order1.begin());

  // Use consequent unique op to get another order_buffer
  //    input2 = 1,1,3,5,5,7,9
  //    order2 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,3,5,7,9
  //    order2 = 0,2,3,5,6
  auto new_last = thrust::unique_by_key(
      thrust::cuda::par.on(context_.cuda_stream()),
      buffer,
      buffer + N,
      order2.begin());
  int K = new_last.first - buffer;

  uniqueTensor->Resize(K);
  T* unique = uniqueTensor->template mutable_data<T>();
  context_.template CopyItems<CUDAContext, CUDAContext>(
      thrust_unique_buffer_.meta(), K, buffer, unique);

  // Compute the remapping. For example, for the number 1, if we look at
  // order2[0] and order2[1], we know that input2[0:2) are all 1. They are all
  // remapped to 0 in final input. And from order1, we know where they come
  // from. The rest is easy.
  if (remapping != nullptr) {
    // record remap
    remap_kernel<<<
        CAFFE_GET_BLOCKS(K),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        order2.data(), order1.data(), remapping, N, K);
  }
}
REGISTER_CUDA_OPERATOR(Unique, UniqueOp<CUDAContext>);
#endif // THRUST_VERSION >= 100800

REGISTER_CUDA_OPERATOR(Size, SizeOp<CUDAContext>);

template <typename T>
__global__ void RangeKernel(const int n, T* Y, T offset, T step) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    Y[index] = index * step + offset;
  }
}

template <>
template <typename T>
bool RangeOp<CUDAContext>::DoRunOnDevice(
    const T& start,
    const T& step,
    Tensor<CUDAContext>* output) {
  int N = output->size();
  RangeKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, output->mutable_data<T>(), start, step);
  return true;
}

REGISTER_CUDA_OPERATOR(Range, RangeOp<CUDAContext>);
} // namespace caffe2
