#include <math.h>
#include <cfloat>
// TODO(jamesreed): I would use <cmath> here but std::isnan
// and std::isinf are declared constexpr there and the nvidia
// compiler throws an error because of it

#include "caffe2/core/context_gpu.h"
#include "utility_ops.h"

namespace caffe2 {

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
    std::cerr << "Tensor contained NaN or inf: " << this->def().input(0)
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
      std::cerr << "Input tensor: " << j << ": [" << def().input(j) << "]"
                << std::endl;
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

template<typename T_INDEX>
__global__ void
GatherKernel(const float* X, float* Y, const T_INDEX* indices, const int N, const int block_size) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = indices[i];
    const float* src_offset = X + idx * block_size;
    float* dst_offset = Y + i   * block_size;
    for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
      dst_offset[j] = src_offset[j];
    }
  }
}

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t,int64_t>>::call(
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

  GatherKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        src_base, out, idxs, N, block_size
      );
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);
}  // namespace
}  // namespace caffe2
