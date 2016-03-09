#include "THCUNN.h"
#include "common.h"

struct MaxFloat
{
  __device__ __forceinline__ float operator()(float max, float v) const
  {
    return fmaxf(max, v);
  }
};

struct SumFloat
{
  __device__ __forceinline__ float operator()(float sum, float v) const
  {
    return sum + v;
  }
};

struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(float v)
    : max_k(v)
  {}

  __device__ __forceinline__ float operator()(float sum, float v) const
  {
    return sum + expf(v - max_k);
  }

  const float max_k;
};

struct NoFinal
{
  __device__ __forceinline__ float operator()(float v) const
  {
    return v;
  }
};

struct LSMFinal
{
  __device__ __forceinline__ LSMFinal(float m)
    : max_k(m)
  {}

  __device__ __forceinline__ float operator()(float v) const
  {
    return max_k + logf(v);
  }

  const float max_k;
};

template <typename Reduction, typename Finalize>
__device__ __forceinline__ float
blockReduce(float* smem, float val,
            const Reduction& r,
            float defaultVal,
            const Finalize& f)
{
  // To avoid RaW races from chaning blockReduce calls together, we
  // need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  float warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if ((threadIdx.x / 32) == 0)
  {
    int lane = threadIdx.x % 32;

    if (lane < blockDim.x / 32)
    {
#pragma unroll
      for (int i = 0; i < 32; ++i)
      {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }

      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  float blockVal = defaultVal;

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < blockDim.x / 32; ++i)
    {
      blockVal = r(blockVal, smem[i]);
    }

    smem[0] = f(blockVal);
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <typename Reduction>
__device__ __forceinline__ float
blockReduce(float* smem, float val,
            const Reduction& r,
            float defaultVal)
{
  return blockReduce<Reduction, NoFinal>(smem, val, r, defaultVal, NoFinal());
}

template <typename Reduction, int ILP>
__device__ __forceinline__ float
ilpReduce(float* data,
          int size,
          const Reduction& r,
          float defaultVal)
{
  float threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP)
  {
    float tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmp[j] = data[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      threadVal = r(threadVal, tmp[j]);
    }
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
  {
    threadVal = r(threadVal, data[offset]);
  }

  return threadVal;
}

template <int ILP>
__global__ void
cunn_LogSoftMax_updateOutput_kernel(float *output, float *input, int classes)
{
  extern __shared__ float buffer[];
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  float threadMax =
    ilpReduce<MaxFloat, ILP>(input, classes, MaxFloat(), -FLT_MAX);
  float max_k =
    blockReduce<MaxFloat>(buffer, threadMax, MaxFloat(), -FLT_MAX);

  float threadExp =
    ilpReduce<SumExpFloat, ILP>(input, classes, SumExpFloat(max_k), 0.0f);
  float logsum_k =
    blockReduce<SumFloat, LSMFinal>(
      buffer, threadExp, SumFloat(), 0.0f, LSMFinal(max_k));

  // Output LSM (hand ILP)
  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    float tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      output[offset + j * blockDim.x] = tmp[j] - logsum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    output[offset] = input[offset] - logsum_k;
  }
}

template <int ILP>
__global__ void
cunn_LogSoftMax_updateGradInput_kernel(float *gradInput,
                                       float *output,
                                       float *gradOutput,
                                       int classes)
{
  extern __shared__ float buffer[];
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  float threadSum =
    ilpReduce<SumFloat, 4>(gradOutput, classes, SumFloat(), 0.0f);
  float sum_k =
    blockReduce<SumFloat>(buffer, threadSum, SumFloat(), 0.0f);

  // Update gradInput (hand ILP)
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    float tmpGradOutput[ILP];
    float tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      gradInput[offset + j * blockDim.x] =
        tmpGradOutput[j] - __expf(tmpOutput[j]) * sum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    gradInput[offset] =
      gradOutput[offset] - __expf(output[offset]) * sum_k;
  }
}

void THNN_CudaLogSoftMax_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);

  int batchSize = 1;
  int classSize = 0;

  if (THCudaTensor_nDimension(state, input) == 1)
  {
    classSize = THCudaTensor_size(state, input, 0);
  }
  else if (THCudaTensor_nDimension(state, input) == 2)
  {
    batchSize = THCudaTensor_size(state, input, 0);
    classSize = THCudaTensor_size(state, input, 1);
  }
  else
  {
    THError("vector or matrix expected");
  }

  dim3 grid(batchSize);
  dim3 block(1024);

  cunn_LogSoftMax_updateOutput_kernel<2>
    <<<grid, block, block.x * sizeof(float), THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, output),
      THCudaTensor_data(state, input),
      classSize
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
  {
    THError(cudaGetErrorString(errcode));
  }

  THCudaTensor_free(state, input);
}

void THNN_CudaLogSoftMax_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  output = THCudaTensor_newContiguous(state, output);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  THCudaTensor_resizeAs(state, gradInput, output);

  int batchSize = 1;
  int classSize = 0;

  if (THCudaTensor_nDimension(state, gradInput) == 1)
  {
    classSize = THCudaTensor_size(state, gradInput, 0);
  }
  else if (THCudaTensor_nDimension(state, gradInput) == 2)
  {
    batchSize = THCudaTensor_size(state, gradInput, 0);
    classSize = THCudaTensor_size(state, gradInput, 1);
  }
  else
  {
    THError("vector or matrix expected");
  }

  dim3 grid(batchSize);
  dim3 block(1024);

  cunn_LogSoftMax_updateGradInput_kernel<2>
    <<<grid, block, block.x * sizeof(float), THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, gradInput),
      THCudaTensor_data(state, output),
      THCudaTensor_data(state, gradOutput),
      classSize
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
  {
    THError(cudaGetErrorString(errcode));
  }

  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, output);
}
